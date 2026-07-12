# KQURA Neural Forge — Serverless (RunPod)

This is the **HD engine** for KQURA 3D Studio, packaged as a **RunPod serverless
endpoint**. Unlike the always-on pod, this:

- costs **$0 when idle** (you pay only per-second while a model is generating),
- **cold-starts itself** the moment someone hits *Forge HD* — no pod to start, no
  tunnel to run, **no terminal ever**,
- uses the **same open-weight engine** (Tencent Hunyuan3D-2) — still KQURA's own
  engine, no third-party 3D generation service.

The finished GLB is POSTed straight back into the studio's asset store (HMAC
authed), so nothing large travels through RunPod's result JSON.

---

## One-time setup (about 20 minutes, mostly waiting on the image build)

### 1. Build & push the image

You need a Docker registry RunPod can pull from (Docker Hub is easiest).

```bash
cd kqura_forge_serverless
docker build -t YOURUSER/kqura-forge:latest .
docker push YOURUSER/kqura-forge:latest
```

> No local GPU or Docker? Use RunPod's built-in **GitHub / image builder**, or run
> the two commands above from any cheap cloud VM. The image is large (~15 GB) because
> it bakes in the Hunyuan3D-2 tooling and the texture-bake components.

### 2. Create the serverless endpoint

RunPod dashboard → **Serverless** → **New Endpoint**:

- **Container image:** `YOURUSER/kqura-forge:latest`
- **GPU:** 24 GB (e.g. RTX 4090 / L4) — enough for shape + texture
- **Active workers:** `0`  (this is what makes it $0 idle)
- **Max workers:** `1` (raise later for concurrency)
- **Idle timeout:** `5` seconds
- **Container disk:** 20 GB; add a **Network Volume** mounted at `/runpod-volume`
  so downloaded model weights persist between cold starts (first generation
  downloads several GB of weights; the volume means it only happens once).

Create it, then copy the **Endpoint ID** (looks like `abc123def456`).

### 3. Get a RunPod API key

RunPod → **Settings → API Keys → +API Key**. Copy it.

### 4. Put both into KQURA

Open **Kqura Admin Hub → Settings** and set:

| Setting | Value |
|---|---|
| `KQURA_FORGE_RUNPOD_ENDPOINT` | the Endpoint ID from step 2 |
| `KQURA_FORGE_RUNPOD_KEY` | the API key from step 3 |
| `KQURA_FORGE_WORKER_SECRET` | any long random string (signs the GLB hand-off) |

Leave `KQURA_FORGE_WORKER_URL` **blank** — that's only for the old self-hosted pod.
The moment the endpoint + key are set, the studio automatically uses serverless for
every *Forge HD* and *Texture* job.

---

## How it works

```
Studio  ── POST https://api.runpod.ai/v2/<ENDPOINT>/run
              { input: { op, prompt, image, views, texture_image, mesh_glb,
                         ingest_url, ingest_key, ingest_token } }
RunPod  ── cold-starts a worker, runs handler.py
handler ── Hunyuan3D-2 sculpt (+ multi-view) + texture  →  finished .glb
handler ── POST ingest_url  (multipart: glb + key + token)  →  studio asset store
handler ── return { key: ingest_key }
Studio  ── polls /status/<id>; on COMPLETED the GLB is already in our store
```

- **Auth:** `ingest_token = HMAC-SHA256(ingest_key, KQURA_FORGE_WORKER_SECRET)`.
  The studio's `ingest` action recomputes it and rejects anything that doesn't match,
  so only jobs the studio actually launched can deposit a file.
- **Backend auto-detect:** if the Hunyuan3D tooling imports, it runs the real HD
  engine; otherwise it falls back to a mock capsule GLB so you can prove the wiring
  before the heavy image finishes building.

## Cost

- Idle: **$0**.
- Per generation: a 4090 worker runs ~$0.0002–0.0004/sec; a textured character is
  typically 60–180 s ⇒ roughly **$0.01–$0.07 each**, plus a one-time cold start
  (~30–60 s) when the endpoint has been asleep.

## Testing

RunPod's endpoint page has a **Requests** tab. Send:

```json
{ "input": { "op": "image3d", "prompt": "test",
  "image": "data:image/png;base64,iVBORw0KGgо...",
  "ingest_url": "https://your-studio-url/KQURA3DStudio.php?__kq3_action=ingest",
  "ingest_key": "kq3_sv_test.glb", "ingest_token": "<hmac you compute>" } }
```

In normal use the studio fills all of this in for you — you never craft it by hand.
