"""Build-time gate for KQURA Engine v2 (Hunyuan3D-2.1).

Installers above are tolerant; THIS decides. Leaf dependencies are checked
diagnostically (named OK/FAIL lines). The build passes or fails ONLY on the
three modules the worker actually runs: the shape engine, the renderer, and
the PBR paint engine. The verdict + reasons print at the END of the log.
"""
import sys
import traceback

REPO = "/app/Hunyuan3D-2.1"
for p in (REPO, REPO + "/hy3dshape", REPO + "/hy3dpaint"):
    sys.path.insert(0, p)

diag = []      # informational failures (leaf deps)
fatal = []     # engine-module failures -> build fails


def check(label, fn, bucket):
    try:
        fn()
        print("OK  :", label)
    except Exception:
        bucket.append(label + ":\n" + traceback.format_exc())
        print("FAIL:", label)


# --- leaf dependencies (diagnostic — they explain WHY an engine check fails) ---
check("torch", lambda: __import__("torch"), diag)
check("torchvision", lambda: __import__("torchvision"), diag)
check("torchvision.transforms.functional_tensor (basicsr needs this)",
      lambda: __import__("torchvision.transforms.functional_tensor"), diag)
check("numpy", lambda: __import__("numpy"), diag)
check("cv2 (opencv)", lambda: __import__("cv2"), diag)
check("diffusers", lambda: __import__("diffusers"), diag)
check("transformers", lambda: __import__("transformers"), diag)
check("omegaconf", lambda: __import__("omegaconf"), diag)
check("einops", lambda: __import__("einops"), diag)
check("trimesh", lambda: __import__("trimesh"), diag)
check("pymeshlab", lambda: __import__("pymeshlab"), diag)
check("pygltflib", lambda: __import__("pygltflib"), diag)
check("xatlas", lambda: __import__("xatlas"), diag)
check("open3d", lambda: __import__("open3d"), diag)
check("basicsr", lambda: __import__("basicsr"), diag)
check("realesrgan", lambda: __import__("realesrgan"), diag)
check("rembg", lambda: __import__("rembg"), diag)
check("custom_rasterizer (compiled)", lambda: __import__("custom_rasterizer"), diag)

try:
    from torchvision_fix import apply_fix
    apply_fix()
    print("OK  : torchvision_fix applied")
except Exception as e:
    print("note: torchvision_fix skipped:", e)

# --- THE ENGINE (decisive) ---
check("ENGINE hy3dshape.pipelines (shape)",
      lambda: __import__("hy3dshape.pipelines", fromlist=["Hunyuan3DDiTFlowMatchingPipeline"]), fatal)
check("ENGINE DifferentiableRenderer.MeshRender (renderer)",
      lambda: __import__("DifferentiableRenderer.MeshRender", fromlist=["MeshRender"]), fatal)
check("ENGINE textureGenPipeline (PBR paint)",
      lambda: __import__("textureGenPipeline"), fatal)

# --- verdict at the END of the log, where screenshots land ---
print("\n==================== GATE VERDICT ====================")
try:
    with open("/tmp/pipskip.log") as f:
        print("packages the installer skipped:\n" + f.read())
except Exception:
    print("packages the installer skipped: (none)")
if diag:
    print("--- leaf-dependency failures (diagnostic) ---")
    for f in diag:
        print(f)
        print("------------------------------------------------------")
if fatal:
    print("--- ENGINE FAILURES (these fail the build) ---")
    for f in fatal:
        print(f)
        print("------------------------------------------------------")
    print("GATE: FAILED — the engine modules above did not import.")
    sys.exit(1)
print("GATE: PASSED — engine v2 imports clean." + (" (some optional leaves failed — see above)" if diag else ""))
