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

# Hunyuan3D-2.1 source + its requirements (strip torch pins: ours stays 2.5.1)
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 /app/Hunyuan3D-2.1 && \
    cd /app/Hunyuan3D-2.1 && \
    sed -i '/^torch\b/d; /^torch=/d; /^torchvision/d; /^torchaudio/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# compiled components (the part that silently failed on v1 — now loud):
# custom_rasterizer (CUDA ext) + DifferentiableRenderer's mesh painter (pybind11)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX" FORCE_CUDA=1
RUN pip install --no-cache-dir ninja pybind11
RUN cd /app/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && \
    ( python setup.py install || pip install --no-cache-dir . )
RUN cd /app/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && \
    ( bash compile_mesh_painter.sh || \
      c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) mesh_inpaint_processor.cpp -o mesh_inpaint_processor$(python3-config --extension-suffix) ) && \
    ls mesh_inpaint_processor*.so
# torch first — the extension links against torch's shared libs (libc10.so)
RUN python -c "import torch, custom_rasterizer; print('KQURA texture-bake (2.1): COMPILED OK')"

# RealESRGAN 4x super-resolution checkpoint (the painter's polish pass)
RUN mkdir -p /app/Hunyuan3D-2.1/hy3dpaint/ckpt && \
    curl -fsSL -o /app/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    python -c "import os; s=os.path.getsize('/app/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth'); assert s > 50_000_000, 'ckpt too small: %d' % s; print('RealESRGAN ckpt OK:', s, 'bytes')"

COPY handler.py /app/handler.py

# RunPod serverless invokes the handler; it never listens on a port.
CMD ["python", "-u", "handler.py"]
