"""
KQURA Neural Forge - RunPod SERVERLESS handler  (ENGINE v2: Hunyuan3D-2.1)
==========================================================================
The quality-leap engine: Tencent Hunyuan3D-2.1 (open weights, KQURA-owned).
vs the 2.0 engine this adds:
  - hunyuan3d-paintpbr-v2-1: PHYSICALLY-BASED texture painting (albedo +
    metallic/roughness) with built-in RealESRGAN 4x SUPER-RESOLUTION —
    the polished-skin look.
  - hunyuan3d-dit-v2-1: newer shape model.

Contract with the studio is UNCHANGED (op/prompt/image/views/texture_image/
mesh_glb/mesh_url + ingest_url/key/token; returns {key} on success, {error}
with traceback tail on failure). Rollback = the *_hy20_backup files.

Env (all optional):
  KQ_FORGE_BACKEND      auto | mock | hy21          (default auto)
  KQ_FORGE_MODEL        tencent/Hunyuan3D-2.1
  KQ_FORGE_PAINT_VIEWS  multiview count for the painter (default 6)
  KQ_FORGE_PAINT_RES    painter view resolution (default 512; 768 = slower/finer)
"""
import base64
import io
import os
import re
import sys
import time
import uuid

import runpod

REPO = "/app/Hunyuan3D-2.1"
# 2.1 ships hy3dshape/ and hy3dpaint/ as separate source trees + expects to run
# from the repo root (its configs/ckpt paths are relative).
for _p in (REPO, os.path.join(REPO, "hy3dshape"), os.path.join(REPO, "hy3dpaint")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
try:
    os.chdir(REPO)
except Exception:
    pass
try:  # 2.1's compatibility shim for newer torchvision
    from torchvision_fix import apply_fix
    apply_fix()
except Exception as _e:
    print("torchvision_fix skipped:", _e)

BACKEND = os.environ.get("KQ_FORGE_BACKEND", "auto")          # auto | mock | hy21
MODEL_ID = os.environ.get("KQ_FORGE_MODEL", "tencent/Hunyuan3D-2.1")
PAINT_VIEWS = int(os.environ.get("KQ_FORGE_PAINT_VIEWS", "6"))
PAINT_RES = int(os.environ.get("KQ_FORGE_PAINT_RES", "512"))
# final baked UV texture resolution — THE crispness ceiling of the exported model.
# Hunyuan defaults this to ~1024; 2048 is sharp + game-ready, 4096 heavier but max.
TEX_SIZE = int(os.environ.get("KQ_FORGE_TEX_SIZE", "2048"))
# auto-enhance uploaded reference images so low-quality phone shots still sculpt &
# paint crisp. Upscales small inputs (RealESRGAN when available, else Lanczos) and
# resamples into the [MIN, MAX] band, then a light unsharp. On by default.
ENHANCE = os.environ.get("KQ_FORGE_ENHANCE", "1") != "0"
ENHANCE_MIN = int(os.environ.get("KQ_FORGE_ENHANCE_MIN", "1024"))
ENHANCE_MAX = int(os.environ.get("KQ_FORGE_ENHANCE_MAX", "2048"))
OUT_DIR = os.environ.get("KQ_FORGE_OUT", "/tmp/kqura_forge_out")
os.makedirs(OUT_DIR, exist_ok=True)

_pipes = {}          # cached across warm invocations
_backend_cached = [None]


def _backend():
    if _backend_cached[0]:
        return _backend_cached[0]
    b = BACKEND
    if b == "auto":
        try:
            import hy3dshape  # noqa: F401
            b = "hy21"
        except Exception as e:
            print("hy3dshape import failed -> mock:", e)
            b = "mock"
    _backend_cached[0] = b
    return b


def _load_pretrained(cls, repo, **kwargs):
    """Self-heal a corrupt/partial HF cache: on load failure wipe THIS model's
    cache dir and re-download once."""
    import shutil, glob
    try:
        return cls.from_pretrained(repo, **kwargs)
    except Exception as e:
        print("model load failed, clearing cache + retrying:", str(e)[:300])
        hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        safe = repo.replace("/", "--")
        for d in glob.glob(os.path.join(hf, "hub", "models--" + safe + "*")):
            shutil.rmtree(d, ignore_errors=True)
        return cls.from_pretrained(repo, **kwargs)


def _shape_pipe():
    if "shape" not in _pipes:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        _pipes["shape"] = _load_pretrained(Hunyuan3DDiTFlowMatchingPipeline, MODEL_ID)
    return _pipes["shape"]


def _paint_pipe():
    if "paint" not in _pipes:
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        def _mk(views, res):
            conf = Hunyuan3DPaintConfig(views, res)
            # raise the FINAL baked texture map size (separate from per-view render
            # res) — this is what makes the exported albedo crisp/HD. Set whichever
            # attribute this build of the config exposes; guarded so it never breaks.
            for _attr in ("texture_size", "tex_resolution", "texture_resolution"):
                if hasattr(conf, _attr):
                    try:
                        setattr(conf, _attr, TEX_SIZE)
                    except Exception:
                        pass
            conf.realesrgan_ckpt_path = os.path.join(REPO, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth")
            conf.multiview_cfg_path = os.path.join(REPO, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
            conf.custom_pipeline = os.path.join(REPO, "hy3dpaint", "hunyuanpaintpbr")
            return Hunyuan3DPaintPipeline(conf)
        try:
            _pipes["paint"] = _mk(PAINT_VIEWS, PAINT_RES)
        except Exception as e:
            print("paint config (%d views, %d res) rejected (%s) -> falling back to 6/%d" % (PAINT_VIEWS, PAINT_RES, str(e)[:120], PAINT_RES))
            _pipes["paint"] = _mk(6, PAINT_RES)
    return _pipes["paint"]


def _decode_image(data_url):
    m = re.match(r"^data:image/(png|jpeg|webp);base64,(.+)$", data_url or "", re.S)
    if not m:
        return None
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(m.group(2)))).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    return bg.convert("RGB")


_rrdb = {}


def _realesrgan_upscale(img):
    """4x AI super-resolution using the RealESRGAN ckpt already in the image.
    Best-effort — raises if the realesrgan/basicsr stack isn't importable, and the
    caller falls back to Lanczos."""
    import numpy as np
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    if "up" not in _rrdb:
        ckpt = os.path.join(REPO, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        _rrdb["up"] = RealESRGANer(scale=4, model_path=ckpt, model=model, tile=512,
                                   tile_pad=10, pre_pad=0, half=True)
    arr = np.array(img.convert("RGB"))[:, :, ::-1]   # RGB -> BGR
    out, _ = _rrdb["up"].enhance(arr, outscale=4)
    from PIL import Image
    return Image.fromarray(out[:, :, ::-1])          # BGR -> RGB


def _enhance_input_image(img):
    """Clean + upscale a user reference so even a blurry, low-res upload becomes a
    crisp reference the sculptor/painter can reproduce faithfully. Never fatal."""
    if img is None or not ENHANCE:
        return img
    try:
        from PIL import Image, ImageFilter
        w, h = img.size
        short = min(w, h)
        if short < ENHANCE_MIN:
            try:
                img = _realesrgan_upscale(img)
            except Exception as e:
                print("input RealESRGAN unavailable, using Lanczos:", str(e)[:100])
        # resample into the target band (preserve aspect), then a light crispen
        w, h = img.size
        short = min(w, h)
        if short > ENHANCE_MAX:
            s = ENHANCE_MAX / float(short)
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        elif short < ENHANCE_MIN:
            s = ENHANCE_MIN / float(short)
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=60, threshold=2))
        return img
    except Exception as e:
        print("input enhance skipped:", str(e)[:120])
        return img


def _strip_base_plate(mesh):
    """Image-to-3D models hallucinate display plates under the feet.
    Pass 1 removes DISCONNECTED plates (flat + wide + at the bottom + small).
    Pass 2 handles plates FUSED to the feet: if the bottom slice of the mesh
    spans most of the footprint in BOTH axes (feet never do — they're wide but
    shallow), slice the mesh just above the plate and drop it back to the
    ground. Conservative: keeps the model if anything looks off."""
    try:
        import trimesh
        b = mesh.bounds
        h = float(b[1][1] - b[0][1]) or 1.0
        w = float(b[1][0] - b[0][0]) or 1.0
        d = float(b[1][2] - b[0][2]) or 1.0
        # pass 1: disconnected plate components
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            total_faces = sum(int(p.faces.shape[0]) for p in parts)
            keep, removed = [], 0
            for p in parts:
                pb = p.bounds
                flat = (pb[1][1] - pb[0][1]) < 0.06 * h
                wide = max(pb[1][0] - pb[0][0], pb[1][2] - pb[0][2]) > 0.45 * max(w, d)
                at_bottom = pb[0][1] <= b[0][1] + 0.03 * h
                small = p.faces.shape[0] < 0.30 * total_faces
                if flat and wide and at_bottom and small:
                    removed += 1
                    continue
                keep.append(p)
            if removed and keep:
                mesh = trimesh.util.concatenate(keep)
                b = mesh.bounds
                print("base plate stripped (%d disconnected component(s))" % removed)
        # pass 2: fused plate — bottom slice covers most of the footprint both ways
        v = mesh.vertices
        slab = v[v[:, 1] < b[0][1] + 0.04 * h]
        if len(slab) > 50:
            sw = float(slab[:, 0].max() - slab[:, 0].min())
            sd = float(slab[:, 2].max() - slab[:, 2].min())
            if sw > 0.55 * w and sd > 0.55 * d:
                cut_y = float(b[0][1] + 0.045 * h)
                cut = mesh.slice_plane([0, cut_y, 0], [0, 1, 0], cap=False)
                if cut is not None and len(cut.faces) > 0.5 * len(mesh.faces):
                    cut.apply_translation([0, b[0][1] - cut.bounds[0][1], 0])
                    print("fused base plate sliced off at y=%.3f" % cut_y)
                    return cut
    except Exception as e:
        print("plate strip skipped:", e)
    return mesh


def _decimate_for_paint(mesh_path):
    """Shrink the raw sculpt (300-600k tris) to a clean paint-ready mesh using
    pymeshlab (installed + green in the gate) — replaces the painter's own
    remesh prep, which needs the perpetually-uninstallable open3d. THIS is the
    wavy/melted-texture fix: painting a decimated mesh gives crisp bakes."""
    target = int(os.environ.get("KQ_FORGE_PAINT_FACES", "40000"))
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh_path)
        n = ms.current_mesh().face_number()
        if n <= target * 1.3:
            return mesh_path
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target, preservenormal=True)
        out = mesh_path + ".dec.obj"
        ms.save_current_mesh(out)
        print("decimated for paint: %d -> ~%d faces" % (n, target))
        return out
    except Exception as e:
        print("decimation skipped (%s) — painting the raw mesh" % str(e)[:150])
        return mesh_path


def _paint_mesh(mesh_path, ref_pil, out_path):
    """Run the 2.1 PBR painter (file-path API) and guarantee a GLB comes out.
    We pre-decimate ourselves (pymeshlab), so the painter's own remesh prep
    (which needs open3d) is skipped when our decimation succeeded."""
    ref_path = out_path + ".ref.png"
    ref_pil.save(ref_path)
    paint_input = _decimate_for_paint(mesh_path)
    pre_decimated = paint_input != mesh_path
    pipe = _paint_pipe()
    try:
        res = pipe(mesh_path=paint_input, image_path=ref_path, output_mesh_path=out_path,
                   use_remesh=not pre_decimated)
    except TypeError:
        res = pipe(paint_input, ref_path, out_path)   # older positional signature
    except Exception as e:
        msg = str(e)
        if "open3d" in msg or "simplify" in msg or "remesh" in msg or "decimation" in msg:
            print("remesh prep failed (%s) -> retrying with use_remesh=False" % msg[:120])
            res = pipe(mesh_path=paint_input, image_path=ref_path, output_mesh_path=out_path, use_remesh=False)
        else:
            raise
    p = res if isinstance(res, str) and os.path.isfile(res) else out_path
    if not os.path.isfile(p):
        raise RuntimeError("painter finished but produced no file")
    with open(p, "rb") as f:
        head = f.read(4)
    if head != b"glTF":   # painter wrote a non-glb container -> repackage
        import trimesh
        m = trimesh.load(p)
        m.export(out_path)
        p = out_path
    return p


def _run_mock(out_path, op, prompt, views, tex):
    import trimesh
    body = trimesh.creation.capsule(radius=0.25, height=0.9)
    body.apply_translation([0, 0.7, 0])
    head = trimesh.creation.icosphere(radius=0.2, subdivisions=3)
    head.apply_translation([0, 1.5, 0])
    trimesh.Scene({"body": body, "head": head}).export(out_path)


def _run_mock_retex(out_path, mesh_glb):
    if mesh_glb:
        with open(out_path, "wb") as f:
            f.write(mesh_glb)
    else:
        _run_mock(out_path, "retex", "", {}, None)


def _shape_pipe_mv():
    """TRUE MULTI-VIEW shape model (Hunyuan3D-2mv, from the 2.0 family): takes
    {front/back/left/right} images so the back/side references actually drive
    the hidden geometry (no more invented tails). The original integration
    failed because this model lives in its OWN repo folder that was never
    specified — from_pretrained needs subfolder='hunyuan3d-dit-v2-mv'."""
    if "shape_mv" not in _pipes:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as MVPipeline
        _pipes["shape_mv"] = _load_pretrained(MVPipeline, "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv")
    return _pipes["shape_mv"]


def _run_hy21(out_path, op, prompt, views, tex, skip_paint=False):
    """front(+back/side) images -> shape -> [PBR texture (+4x SR)] -> GLB.
    Multi-view (>=2 views) uses the dedicated 2mv shape model so hidden parts
    follow YOUR references; single view uses the 2.1 shape model. With
    skip_paint the gray master ships as-is (two-step Meshy-style flow)."""
    if not views or "front" not in views:
        raise RuntimeError("This engine sculpts from an image; give it at least a front view.")
    from hy3dshape.rembg import BackgroundRemover
    rem = BackgroundRemover()
    clean = {}
    for k, im in views.items():
        try:
            clean[k] = rem(im)
        except Exception:
            clean[k] = im
    front = clean["front"]

    # max-fidelity knobs (env-tunable, safe fallback if a pipeline rejects them)
    _steps = int(os.environ.get("KQ_FORGE_SHAPE_STEPS", "50"))
    _octree = int(os.environ.get("KQ_FORGE_OCTREE", "384"))

    def _sculpt(pipe, image):
        try:
            return pipe(image=image, num_inference_steps=_steps, octree_resolution=_octree)[0]
        except TypeError:
            return pipe(image=image)[0]

    mesh = None
    if len(clean) >= 2:
        try:
            mesh = _sculpt(_shape_pipe_mv(), clean)
            print("multi-view sculpt used (%d views)" % len(clean))
        except Exception:
            import traceback
            print("multi-view sculpt failed, falling back to single view:", traceback.format_exc()[-600:])
            mesh = None
    if mesh is None:
        mesh = _sculpt(_shape_pipe(), front)
    mesh = _strip_base_plate(mesh)

    if skip_paint:
        # gray master: ship with a proper clay PBR material (raw sculpts carry
        # dark/absent visuals -> the black-silhouette bug) and real normals
        try:
            from trimesh.visual import TextureVisuals
            from trimesh.visual.material import PBRMaterial
            mesh.visual = TextureVisuals(material=PBRMaterial(
                baseColorFactor=[0.72, 0.75, 0.80, 1.0], metallicFactor=0.0, roughnessFactor=0.9))
            _ = mesh.vertex_normals   # force normals into the export
        except Exception as e:
            print("clay material skipped:", e)
        mesh.export(out_path)
        return

    raw_path = out_path + ".raw.glb"
    mesh.export(raw_path)
    ref = tex if tex is not None else views["front"]
    if tex is not None:
        try:
            ref = rem(tex)
        except Exception:
            ref = tex
    try:
        _paint_mesh(raw_path, ref, out_path)
    except Exception as e:
        import traceback
        print("PBR texture stage failed, delivering shape-only:", traceback.format_exc()[-800:])
        import shutil
        shutil.copyfile(raw_path, out_path)
    finally:
        try:
            os.remove(raw_path)
        except OSError:
            pass


def _run_retex21(out_path, mesh_glb, tex_img, front_img):
    ref = tex_img if tex_img is not None else front_img
    if not mesh_glb:
        raise RuntimeError("retex needs the model GLB")
    if ref is None:
        raise RuntimeError("retex needs a reference image")
    try:
        from hy3dshape.rembg import BackgroundRemover
        ref = BackgroundRemover()(ref)
    except Exception:
        pass
    raw_path = out_path + ".raw.glb"
    with open(raw_path, "wb") as f:
        f.write(mesh_glb)
    try:
        _paint_mesh(raw_path, ref, out_path)
    finally:
        try:
            os.remove(raw_path)
        except OSError:
            pass


def _run_autorig(out_path, mesh_glb, joints):
    """PRO RIG: Blender's bone-heat automatic weights — the same class of
    volumetric weighting Mixamo/Meshy-grade rigs use (heat can't jump across
    air, so a hand never grabs the opposite leg). The studio sends the joint
    positions the user placed/adjusted; Blender builds the armature and solves
    the weights; we ship back a skinned GLB."""
    import bpy
    from mathutils import Vector
    raw = out_path + ".in.glb"
    with open(raw, "wb") as f:
        f.write(mesh_glb)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=raw)

    # STRIP any rig the incoming model already carried. If the studio sent an
    # already-rigged version (e.g. a prior KQURA/Blender rig), its armature +
    # vertex groups would survive the import and collide with the fresh skeleton
    # we build below — the glTF exporter would then emit a DOUBLED skin (two
    # armatures fighting over the same verts) and the mesh explodes on any pose.
    # Reduce to a clean, naked mesh first so ARMATURE_AUTO binds exactly our bones.
    for o in list(bpy.data.objects):
        if o.type == "ARMATURE":
            bpy.data.objects.remove(o, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no meshes found in the model")
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    # drop leftover armature modifiers, old vertex groups, and any parenting so
    # the fresh bone-heat solve starts from a blank slate (this is what keeps the
    # exported skin at exactly our joint count instead of old+new merged)
    for mod in list(mesh.modifiers):
        if mod.type == "ARMATURE":
            mesh.modifiers.remove(mod)
    try:
        mesh.vertex_groups.clear()
    except Exception:
        for vg in list(mesh.vertex_groups):
            mesh.vertex_groups.remove(vg)
    try:
        mesh.parent = None
    except Exception:
        pass

    # BAKE the glTF-importer's Y-up->Z-up rotation into the mesh's vertices so no
    # residual object rotation survives. Blender's importer stands a Y-up model up
    # by putting a +90deg X rotation on the mesh OBJECT (not the verts); if we leave
    # it, the exporter writes that +90deg back into the model node and the Y-up
    # studio renders the rigged model lying on its back / upside-down. Applying the
    # transform makes the object matrix identity with standing Z-up verts, so the
    # single export conversion lands it upright.
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception as e:
        print("transform_apply skipped:", e)

    # three.js is Y-up; Blender is Z-up (the glTF importer converts the mesh the
    # same way): (x, y, z) -> (x, -z, y)
    conv = lambda p: Vector((float(p[0]), -float(p[2]), float(p[1])))
    arm = bpy.data.armatures.new("KQRig")
    armobj = bpy.data.objects.new("KQRig", arm)
    bpy.context.collection.objects.link(armobj)
    bpy.context.view_layer.objects.active = armobj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    by_name = {j["name"]: j for j in joints}
    kids = {}
    for j in joints:
        kids.setdefault(j.get("parent") or "", []).append(j)
    for j in joints:
        b = arm.edit_bones.new(j["name"])
        b.head = conv(j["head"])
        ch = kids.get(j["name"]) or []
        if ch:
            b.tail = conv(ch[0]["head"])
        else:
            b.tail = b.head + Vector((0, 0, 0.06))
        if (b.tail - b.head).length < 1e-4:
            b.tail = b.head + Vector((0, 0, 0.06))
        eb[j["name"]] = b
    for j in joints:
        p = j.get("parent")
        if p and p in eb:
            eb[j["name"]].parent = eb[p]
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    armobj.select_set(True)
    bpy.context.view_layer.objects.active = armobj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")   # <- the bone-heat solve
    bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB")
    try:
        os.remove(raw)
    except OSError:
        pass


def _ingest(ingest_url, key, token, glb_path):
    import urllib.request
    with open(glb_path, "rb") as f:
        glb = f.read()
    if not glb.startswith(b"glTF"):
        raise RuntimeError("generated file is not a valid GLB")
    boundary = "----kqura" + uuid.uuid4().hex
    parts = []

    def _field(name, value):
        parts.append(("--" + boundary).encode())
        parts.append(('Content-Disposition: form-data; name="%s"' % name).encode())
        parts.append(b"")
        parts.append(str(value).encode())

    _field("key", key)
    _field("token", token)
    parts.append(("--" + boundary).encode())
    parts.append(('Content-Disposition: form-data; name="glb"; filename="%s"' % key).encode())
    parts.append(b"Content-Type: model/gltf-binary")
    parts.append(b"")
    body = b"\r\n".join(parts) + b"\r\n" + glb + ("\r\n--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(
        ingest_url, data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary,
                 "Content-Length": str(len(body))})
    resp = urllib.request.urlopen(req, timeout=120).read()
    return resp.decode("utf-8", "replace")


def handler(event):
    inp = event.get("input") or {}
    op = inp.get("op", "image3d")
    prompt = (inp.get("prompt") or "")[:2000]

    ingest_url = inp.get("ingest_url") or ""
    ingest_key = inp.get("ingest_key") or ""
    ingest_token = inp.get("ingest_token") or ""
    if not (ingest_url and ingest_key and ingest_token):
        return {"error": "missing ingest_url / ingest_key / ingest_token"}

    views = {}
    raw_views = inp.get("views") or {}
    if isinstance(raw_views, dict):
        for k in ("front", "back", "left", "right"):
            im = _decode_image(raw_views.get(k) or "")
            if im is not None:
                views[k] = _enhance_input_image(im)
    if "front" not in views:
        im = _decode_image(inp.get("image") or "")
        if im is not None:
            views["front"] = _enhance_input_image(im)
    tex = _enhance_input_image(_decode_image(inp.get("texture_image") or ""))

    mesh_glb = None
    mg = inp.get("mesh_glb") or ""
    if mg:
        if mg.startswith("data:"):
            mg = mg.split(",", 1)[-1]
        try:
            mesh_glb = base64.b64decode(mg)
        except Exception:
            mesh_glb = None
    # big meshes ride as a signed download URL (RunPod caps /run at ~10MB)
    if mesh_glb is None and (inp.get("mesh_url") or ""):
        try:
            import urllib.request
            req = urllib.request.Request(inp["mesh_url"], headers={"User-Agent": "KQURA-Forge/2.1"})
            data = urllib.request.urlopen(req, timeout=180).read()
            if data.startswith(b"glTF"):
                mesh_glb = data
            else:
                return {"error": "mesh_url did not return a GLB (got %d bytes, wrong magic)" % len(data)}
        except Exception as e:
            return {"error": "could not download the model to texture: " + str(e)[:300]}

    if op == "image3d" and "front" not in views:
        return {"error": "image3d needs at least a front image"}
    if op == "retex" and (mesh_glb is None or (tex is None and "front" not in views)):
        return {"error": "retex needs mesh_glb and a reference image"}
    if op == "autorig" and (mesh_glb is None or not inp.get("joints")):
        return {"error": "autorig needs the model mesh and the joint list"}

    out_path = os.path.join(OUT_DIR, uuid.uuid4().hex[:20] + ".glb")
    t0 = time.time()
    try:
        if op == "autorig":
            _run_autorig(out_path, mesh_glb, inp.get("joints") or [])
        elif op == "retex":
            front = views.get("front")
            if _backend() == "hy21":
                _run_retex21(out_path, mesh_glb, tex, front)
            else:
                _run_mock_retex(out_path, mesh_glb)
        elif _backend() == "hy21":
            _run_hy21(out_path, op, prompt, views, tex, skip_paint=bool(inp.get("skip_paint")))
        else:
            _run_mock(out_path, op, prompt, views, tex)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return {"error": (str(e) or "generation failed")[:300] + "  ||  " + tb[-700:]}

    if not os.path.isfile(out_path) or os.path.getsize(out_path) < 40:
        return {"error": "generation produced no model file"}

    try:
        ingest_resp = _ingest(ingest_url, ingest_key, ingest_token, out_path)
    except Exception as e:
        return {"error": "ingest failed: " + str(e)[:400]}
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass

    return {"key": ingest_key, "backend": _backend(), "engine": "hunyuan3d-2.1-pbr",
            "seconds": round(time.time() - t0, 1), "ingest": ingest_resp[:200]}


runpod.serverless.start({"handler": handler})
