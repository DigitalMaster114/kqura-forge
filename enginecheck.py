"""Build-time gate for KQURA Engine v2 (Hunyuan3D-2.1).

The requirements install is tolerant (demo-only extras may be skipped), so THIS
is the loud check that matters: if the actual modules the worker uses don't
import, the build fails HERE with the real reason printed.
"""
import sys
import traceback

REPO = "/app/Hunyuan3D-2.1"
for p in (REPO, REPO + "/hy3dshape", REPO + "/hy3dpaint"):
    sys.path.insert(0, p)

failures = []

try:
    import torch
    print("torch:", torch.__version__)
except Exception:
    failures.append("torch:\n" + traceback.format_exc())

try:
    import custom_rasterizer  # noqa: F401
    print("custom_rasterizer: OK")
except Exception:
    failures.append("custom_rasterizer:\n" + traceback.format_exc())

try:
    from torchvision_fix import apply_fix
    apply_fix()
    print("torchvision_fix: applied")
except Exception as e:
    print("torchvision_fix: skipped:", e)   # non-fatal; may live elsewhere

try:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
    print("hy3dshape (shape engine): OK")
except Exception:
    failures.append("hy3dshape:\n" + traceback.format_exc())

try:
    import textureGenPipeline  # noqa: F401
    print("hy3dpaint (PBR texture engine): OK")
except Exception:
    failures.append("hy3dpaint/textureGenPipeline:\n" + traceback.format_exc())

if failures:
    print("\n================ ENGINE GATE FAILED ================")
    for f in failures:
        print(f)
    sys.exit(1)
print("\nKQURA engine v2 gate: ALL IMPORTS OK")
