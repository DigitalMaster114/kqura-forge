# KQURA Neural Forge - RunPod serverless image (Hunyuan3D-2, open weights)
# Build & push once, then point a RunPod serverless endpoint at the image.
# PyTorch 2.4 (CUDA 12.4): new enough for the Hunyuan3D/diffusers/accelerate
# stack (which references torch.xpu, added in torch 2.4), and runs on Ampere
# (A6000/A40/A100) and Ada GPUs.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    KQ_FORGE_BACKEND=auto \
    KQ_FORGE_MODEL=tencent/Hunyuan3D-2 \
    KQ_FORGE_MODEL_MV=tencent/Hunyuan3D-2mv \
    HF_HOME=/opt/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    KQ_FORGE_OUT=/tmp/kqura_forge_out

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# service + serverless deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Hunyuan3D-2 (open HD model tooling) + the texture-bake compiled components.
# Strip any torch/torchvision pins from its requirements so the base image's
# torch 2.4 (which has torch.xpu) is kept — otherwise a downgrade to 2.2 brings
# back "module 'torch' has no attribute 'xpu'". custom_rasterizer then compiles
# against the final torch 2.4.
RUN git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2 /app/Hunyuan3D-2 && \
    cd /app/Hunyuan3D-2 && \
    sed -i '/^torch\b/d; /^torchvision\b/d; /^torchaudio\b/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e . && \
    ( cd hy3dgen/texgen/custom_rasterizer && python setup.py install ) && \
    ( cd hy3dgen/texgen/differentiable_renderer && python setup.py install ) && \
    python -c "import custom_rasterizer; print('texture-bake: ON')" || echo "texture-bake: OFF"

# The texture (paint) pipeline imports diffusers' StableDiffusion + AutoencoderKL.
# These four packages MUST agree or the import cascades through version errors:
#   torch.xpu missing            -> need torch >= 2.4 (base image has it)
#   infer_schema: param q ...    -> diffusers too new/old for torch
#   FLAX_WEIGHTS_NAME missing    -> transformers too new for diffusers 0.30
# This is a single coherent, known-good snapshot (Hunyuan3D-2's mid-2024 target)
# pinned together so nothing floats to an incompatible latest. Installed LAST so
# it wins over whatever Hunyuan's requirements pulled.
RUN pip install --no-cache-dir \
      "diffusers==0.30.0" \
      "transformers==4.44.2" \
      "huggingface_hub==0.24.6" \
      "accelerate==0.33.0" \
      "tokenizers>=0.19,<0.20" \
      "peft==0.12.0"
RUN python -c "from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline; from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL; print('KQURA texture stack import: OK')" \
    || echo "WARN: texture stack still not importable — check the pinned versions"

# faster, more resilient model downloads at runtime (parallel, resumable)
RUN pip install --no-cache-dir "hf_transfer>=0.1.6"

COPY handler.py /app/handler.py

# RunPod serverless invokes the handler; it never listens on a port.
CMD ["python", "-u", "handler.py"]
