# KQURA Neural Forge - RunPod serverless image
# ENGINE v2: Hunyuan3D-2.1 (open weights) — PBR texture painter with built-in
# RealESRGAN 4x super-resolution = the texture-quality leap.
# Rollback: Dockerfile_hy20_backup.txt + handler_hy20_backup.py (rename & re-upload).
#
# Lessons already baked in (learned the hard way on v1):
#  - build machine has NO GPU -> TORCH_CUDA_ARCH_LIST + FORCE_CUDA for the
#    compiled CUDA extension, and verification imports torch FIRST (libc10.so)
#  - every required step FAILS THE BUILD loudly — no `|| echo` on anything vital
#  - strip torch pins from upstream requirements so they can't downgrade ours
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    KQ_FORGE_BACKEND=auto \
    KQ_FORGE_MODEL=tencent/Hunyuan3D-2.1 \
    HF_HOME=/opt/hf \
    KQ_FORGE_OUT=/tmp/kqura_forge_out

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential curl libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# service + serverless deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# torch 2.5.1 + cu124: Hunyuan3D-2.1's pinned target
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# --- Hunyuan3D-2.1 source (separate step so a clone failure is unmistakable) ---
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 /app/Hunyuan3D-2.1 \
 || git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2.1 /app/Hunyuan3D-2.1
RUN ls /app/Hunyuan3D-2.1/hy3dshape /app/Hunyuan3D-2.1/hy3dpaint

# --- its requirements, TOLERANTLY ---
# Strip torch pins (ours stays 2.5.1) and demo-only extras (bpy/gradio/etc — the
# blender addon + web demo, not the engine; some have no wheels for this python).
# Then install LINE BY LINE, logging skips instead of dying: the enginecheck
# gate below is the real pass/fail, and it names exactly what's missing.
RUN cd /app/Hunyuan3D-2.1 && \
    sed -i '/^torch\b/d; /^torch=/d; /^torch>/d; /^torchvision/d; /^torchaudio/d' requirements.txt && \
    grep -viE '^(bpy|gradio|streamlit|jupyter|notebook|sentry)' requirements.txt > /tmp/req.txt && \
    echo '--- installing ---' && cat /tmp/req.txt
RUN set +e; while IFS= read -r pkg; do \
      pkg="$(echo "$pkg" | tr -d '\r')"; \
      [ -z "$pkg" ] && continue; case "$pkg" in \#*) continue;; esac; \
      pip install --no-cache-dir "$pkg" || echo "SKIPPED: $pkg" >> /tmp/pipskip.log; \
    done < /tmp/req.txt; \
    echo '--- skipped packages (if any) ---'; cat /tmp/pipskip.log 2>/dev/null || echo '(none)'; exit 0

# --- the CRITICAL runtime set, pinned to 2.1's own requirements — LOUD.
# The tolerant loop above may skip things; these are the packages the engine
# demonstrably imports (verified against the repo source). If one of these
# fails, the build fails HERE naming it.
RUN pip install --no-cache-dir \
      "transformers==4.46.0" "diffusers==0.30.0" "accelerate==1.1.1" \
      "pytorch-lightning==1.9.5" "omegaconf==2.3.0" "einops==0.8.0" \
      "opencv-python==4.10.0.84" "realesrgan==0.3.0" "basicsr==1.4.2" \
      "rembg==2.0.65" "onnxruntime==1.16.3" pymeshlab "pygltflib==1.16.3" \
      "xatlas==0.0.9" "open3d==0.18.0" "scikit-image==0.24.0" "imageio==2.36.0" \
      timm "safetensors==0.4.4" "huggingface-hub==0.30.2" && \
    pip install --no-cache-dir "numpy==1.24.4"

# torchvision >= 0.17 removed transforms.functional_tensor, but basicsr (used by
# the RealESRGAN super-resolution pass) still imports it. Write a permanent shim
# so that import always works — file-level, no runtime monkeypatching needed.
RUN python -c "import torchvision.transforms.functional_tensor" 2>/dev/null || ( \
      TVDIR="$(python -c 'import torchvision, os; print(os.path.dirname(torchvision.__file__))')" && \
      printf 'from torchvision.transforms.functional import *  # KQURA shim: module removed in torchvision>=0.17\n' > "${TVDIR}/transforms/functional_tensor.py" && \
      python -c "import torchvision.transforms.functional_tensor; print('functional_tensor shim OK')" )

# --- compiled components (loud) ---
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX" FORCE_CUDA=1
RUN pip install --no-cache-dir ninja pybind11
RUN cd /app/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && \
    ( python setup.py install || pip install --no-cache-dir . )
# Their compile_mesh_painter.sh names the output via `python3-config`, which is
# NOT on this image's PATH -> empty suffix -> module unimportable (that was the
# exit-2 failure). Compute the suffix with python itself instead — always present.
RUN cd /app/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && \
    SUF="$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')" && \
    echo "extension suffix: ${SUF}" && \
    c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) mesh_inpaint_processor.cpp -o "mesh_inpaint_processor${SUF}" && \
    ls -la mesh_inpaint_processor*

# --- THE GATE: the modules the worker actually uses must import, or fail loud ---
COPY enginecheck.py /app/enginecheck.py
RUN python /app/enginecheck.py

# RealESRGAN 4x super-resolution checkpoint (the painter's polish pass)
RUN mkdir -p /app/Hunyuan3D-2.1/hy3dpaint/ckpt && \
    curl -fsSL -o /app/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    python -c "import os; s=os.path.getsize('/app/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth'); assert s > 50_000_000, 'ckpt too small: %d' % s; print('RealESRGAN ckpt OK:', s, 'bytes')"

COPY handler.py /app/handler.py

# RunPod serverless invokes the handler; it never listens on a port.
CMD ["python", "-u", "handler.py"]
