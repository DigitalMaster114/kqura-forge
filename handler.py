"""
KQURA Neural Forge - RunPod SERVERLESS handler
==============================================
Same neural engine as the self-hosted worker (Tencent Hunyuan3D-2, open
weights - no third-party 3D services), packaged as a RunPod *serverless*
endpoint so it costs $0 when idle and cold-starts itself the instant the
studio sends a job. No always-on pod, no tunnel, no terminal.

Flow:
  studio  --POST /run { input:{op, prompt, image, views, texture_image,
                                mesh_glb, ingest_url, ingest_key,
                                ingest_token} }-->  this handler
  handler --sculpt/retex--> finished GLB
  handler --POST ingest_url (multipart glb + key + token)--> studio asset store
  handler --return { key: ingest_key }--> RunPod --> studio /status COMPLETED

The studio already knows the destination asset key up front, so the moment
RunPod reports COMPLETED the file is already sitting in our store. The GLB
never travels back through RunPod's result JSON (no size limit, no S3).

Env (set on the RunPod endpoint, optional):
  KQ_FORGE_BACKEND   auto | mock | hunyuan3d   (default auto)
  KQ_FORGE_MODEL     tencent/Hunyuan3D-2       (single-view shape model)
  KQ_FORGE_MODEL_MV  tencent/Hunyuan3D-2mv     (multi-view shape model)
"""
import base64
import io
import os
import re
import time
import uuid

import runpod

BACKEND = os.environ.get("KQ_FORGE_BACKEND", "auto")            # auto | mock | hunyuan3d
MODEL_ID = os.environ.get("KQ_FORGE_MODEL", "tencent/Hunyuan3D-2")
MV_MODEL = os.environ.get("KQ_FORGE_MODEL_MV", "tencent/Hunyuan3D-2mv")
OUT_DIR = os.environ.get("KQ_FORGE_OUT", "/tmp/kqura_forge_out")
os.makedirs(OUT_DIR, exist_ok=True)

# pipelines are cached across warm invocations so only the first (cold) job
# pays the model-load cost.
_hy_pipes = {}
_backend_cached = [None]


def _backend():
    if _backend_cached[0]:
        return _backend_cached[0]
    b = BACKEND
    if b == "auto":
        try:
            import hy3dgen  # noqa: F401
            b = "hunyuan3d"
        except Exception:
            b = "mock"
    _backend_cached[0] = b
    return b


def _load_pretrained(cls, repo, **kwargs):
    """Load a HF pipeline, self-healing a corrupt/partial cache. If the first
    load fails (e.g. "Something wrong while loading .../snapshots/..." from a
    download that was interrupted), wipe THIS model's cache dir and re-download
    once. This kills the class of runtime failures we can't bake around."""
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


def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _ensure_texture_models():
    """Pre-download the paint pipeline's two sub-models (delight + paint-turbo)
    EXPLICITLY. Hunyuan's own loader swallows download errors behind a generic
    "Something wrong while loading ..." — doing it ourselves surfaces the REAL
    cause (usually HF rate-limiting without a token) and, with HF_TOKEN set,
    just works. Cached, so it's a no-op once present."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        "tencent/Hunyuan3D-2",
        allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0/*", "hunyuan3d-paint-v2-0-turbo/*"],
        token=_hf_token(), max_workers=4,
    )


def _decode_image(data_url):
    m = re.match(r"^data:image/(png|jpeg|webp);base64,(.+)$", data_url or "", re.S)
    if not m:
        return None
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(m.group(2)))).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    return bg.convert("RGB")


def _run_mock(out_path, op, prompt, views, tex):
    """No-GPU placeholder GLB so the whole studio->serverless->studio loop
    can be verified before the model weights are installed."""
    import trimesh
    body = trimesh.creation.capsule(radius=0.25, height=0.9)
    body.apply_translation([0, 0.7, 0])
    head = trimesh.creation.icosphere(radius=0.2, subdivisions=3)
    head.apply_translation([0, 1.5, 0])
    trimesh.Scene({"body": body, "head": head}).export(out_path)


def _run_mock_retex(out_path, mesh_glb):
    if mesh_glb:
        with open(out_path, "wb") as f:
            f.write(mesh_glb)   # echo the mesh back so the studio flow completes
    else:
        _run_mock(out_path, "retex", "", {}, None)


def _run_hunyuan(out_path, op, prompt, views, tex):
    """Hunyuan3D-2: image(s) -> shape -> texture -> GLB.
    views: {front/back/left/right: PIL} (front required). Multi-view uses the
    back/side references so the model stops inventing hidden geometry."""
    if not views or "front" not in views:
        raise RuntimeError("This backend sculpts from an image; give it at least a front view.")
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    rem = BackgroundRemover()
    clean = {}
    for k, im in views.items():
        try:
            clean[k] = rem(im)
        except Exception:
            clean[k] = im
    front = clean["front"]

    mesh = None
    if len(clean) >= 2:
        try:
            if "shape_mv" not in _hy_pipes:
                _hy_pipes["shape_mv"] = _load_pretrained(Hunyuan3DDiTFlowMatchingPipeline, MV_MODEL)
            mesh = _hy_pipes["shape_mv"](image=clean)[0]
        except Exception as e:
            print("multiview failed, falling back to single view:", e)
            mesh = None
    if mesh is None:
        if "shape" not in _hy_pipes:
            _hy_pipes["shape"] = _load_pretrained(Hunyuan3DDiTFlowMatchingPipeline, MODEL_ID)
        mesh = _hy_pipes["shape"](image=front)[0]

    tex_img = tex if tex is not None else front
    if tex is not None:
        try:
            tex_img = rem(tex)
        except Exception:
            tex_img = tex
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        if "paint" not in _hy_pipes:
            _ensure_texture_models()
            _hy_pipes["paint"] = _load_pretrained(Hunyuan3DPaintPipeline, "tencent/Hunyuan3D-2")
        mesh = _hy_pipes["paint"](mesh, image=tex_img)
    except Exception as e:  # texture stage optional (needs more VRAM)
        print("texture stage skipped:", e)

    mesh.export(out_path)


def _run_retex(out_path, mesh_glb, tex_img, front_img):
    """Re-texture an EXISTING mesh (GLB bytes) from a reference image."""
    import trimesh
    if not mesh_glb:
        raise RuntimeError("retex needs the model GLB")
    ref = tex_img if tex_img is not None else front_img
    if ref is None:
        raise RuntimeError("retex needs a reference image")
    scene = trimesh.load(io.BytesIO(mesh_glb), file_type="glb")
    mesh = scene.dump(concatenate=True) if hasattr(scene, "dump") else scene
    try:
        from hy3dgen.rembg import BackgroundRemover
        ref = BackgroundRemover()(ref)
    except Exception:
        pass
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    if "paint" not in _hy_pipes:
        _ensure_texture_models()
        _hy_pipes["paint"] = _load_pretrained(Hunyuan3DPaintPipeline, "tencent/Hunyuan3D-2")
    mesh = _hy_pipes["paint"](mesh, image=ref)
    mesh.export(out_path)


def _ingest(ingest_url, key, token, glb_path):
    """POST the finished GLB back to the studio's asset store (HMAC authed)."""
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
                views[k] = im
    if "front" not in views:
        im = _decode_image(inp.get("image") or "")
        if im is not None:
            views["front"] = im
    tex = _decode_image(inp.get("texture_image") or "")

    mesh_glb = None
    mg = inp.get("mesh_glb") or ""
    if mg:
        if mg.startswith("data:"):
            mg = mg.split(",", 1)[-1]
        try:
            mesh_glb = base64.b64decode(mg)
        except Exception:
            mesh_glb = None

    if op == "image3d" and "front" not in views:
        return {"error": "image3d needs at least a front image"}
    if op == "retex" and (mesh_glb is None or (tex is None and "front" not in views)):
        return {"error": "retex needs mesh_glb and a reference image"}

    out_path = os.path.join(OUT_DIR, uuid.uuid4().hex[:20] + ".glb")
    t0 = time.time()
    try:
        if op == "retex":
            front = views.get("front")
            if _backend() == "hunyuan3d":
                _run_retex(out_path, mesh_glb, tex, front)
            else:
                _run_mock_retex(out_path, mesh_glb)
        elif _backend() == "hunyuan3d":
            _run_hunyuan(out_path, op, prompt, views, tex)
        else:
            _run_mock(out_path, op, prompt, views, tex)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)   # full trace in the RunPod logs
        # surface the real cause (last frames) instead of a swallowed generic message
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

    return {"key": ingest_key, "backend": _backend(),
            "seconds": round(time.time() - t0, 1), "ingest": ingest_resp[:200]}


runpod.serverless.start({"handler": handler})
