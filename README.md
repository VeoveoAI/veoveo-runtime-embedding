# veoveo-runtime-embedding

Private bounded EmbeddingGemma 2 inference behind `veoveo-mcp-embedding`, using upstream Sentence Transformers and PyTorch. This component owns GPU execution; the MCP service owns Hub identity, permissions and usage accounting. Read [DESIGN.md](DESIGN.md) for the runtime contract and the [architecture](https://github.com/VeoveoAI/veoveo-docs/blob/main/LORE_OPENSHELL_INTEGRATION.md#embedding-generation) for platform responsibilities.

Mac MPS and Linux CUDA deployment qualification passed, including instrumented GPU forwards, real Hub inference, cross-device numerical parity, capacity and restart persistence. The [execution plan](https://github.com/VeoveoAI/veoveo-docs/blob/main/OPENSHELL_HARDCUT_PLAN.md#embedding-generation--2026-10-19) records source identities, measurements and proof limits. Synthetic component tests are reported separately.

The fixed model is `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b`. Outputs are ordered, normalized 768-dimensional FP32 vectors. Targets are native macOS/arm64 MPS and Linux/amd64 CUDA 13 through OpenShell CDI. CPU, FP16, user-selected models and truncation are unsupported. No texts/vectors are stored, and there is no search API.

## Deterministic checks

Use uv **0.13.1** and Python **3.14.6**. The checked-in lock pins packages and their selected indexes. Base dependencies contain no model packages; these checks perform no model inference or downloads:

```sh
uv sync --frozen --no-dev
uv run --frozen --no-sync python -m unittest discover -s tests -v
docker build --target check --progress=plain .
docker build --platform linux/amd64 --target check --progress=plain .
```

The Docker check supports Linux/arm64 on a Mac and Linux/amd64. Synthetic preprocessing/model-seam/admission fixtures test behavior; their vectors and token counts are not real embeddings or tokenizer evidence. Full checks/builds belong to the builder during coordinated work.

## Cache preparation

Before serving, a trusted operator installs the model extra and prefetches the exact snapshot into the canonical Hugging Face cache. Prefetch is the only network download step; it is independent of requests. Choose a persistent private cache directory, then make the cache read-only to the runtime identity. No custom weights manifest is required.

```sh
uv sync --frozen --no-dev --extra model
export HF_HOME=/var/lib/veoveo/embedding/huggingface
uv run --frozen --no-sync python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="google/embeddinggemma-2", revision="914f7f89142e33e77833254d9c9b90c3cef7303b")'
```

Run prefetch without `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` set. Serving sets both to `1` and loads the canonical repo/revision with `local_files_only=True`. A missing/incompatible snapshot fails startup; requests cannot trigger a download or choose a path. Record source/image identity and independently hash snapshot files in qualification evidence. Serving checks GPU availability/FP32 parameter placement and explicitly selects the native encode device; actual forward-placement instrumentation belongs to separate qualification probes, without hooks or upstream method wrappers in the serving process. A malicious operator modifying the trusted cache is outside this boundary.

## Native macOS process

Install the reviewed source and its virtual environment in an operator-chosen persistent directory. `deploy/ai.veoveo.runtime.embedding.plist` is a native launchd template, with configurable absolute paths for interpreter, working directory, private token and cache. Replace its source-commit placeholder and paths before installing it in the user's `~/Library/LaunchAgents/`. It intentionally contains no credential value.

Create a random private bearer in a mode-0600 file readable by the runtime user. The admitted OpenShell provider must use the same secret through native credential custody; never put it in a repository, CLI argument or shared log. Required startup configuration:

| Variable | Meaning |
|---|---|
| `VEOVEO_EMBEDDING_TOKEN_FILE` | Private bearer file; no group/other permission bits |
| `VEOVEO_RUNTIME_REVISION` | Reviewed 40-character source commit |
| `VEOVEO_EMBEDDING_DEVICE` | Explicit `mps` on Mac or `cuda` on Linux |
| `HF_HOME` | Trusted prefetched canonical cache |
| `VEOVEO_EMBEDDING_BIND` | Defaults to `127.0.0.1`; no public listener |
| `VEOVEO_EMBEDDING_PORT` | Defaults to `8804` |

```sh
VEOVEO_EMBEDDING_DEVICE=mps VEOVEO_RUNTIME_REVISION=<reviewed-commit> VEOVEO_EMBEDDING_TOKEN_FILE=<private-file> HF_HOME=<cache-directory> .venv/bin/python -m veoveo_runtime_embedding
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.veoveo.runtime.embedding.plist
launchctl kickstart gui/$(id -u)/ai.veoveo.runtime.embedding
launchctl bootout gui/$(id -u)/ai.veoveo.runtime.embedding
```

Use either the direct foreground command for an operator probe or launchd for the managed process. First qualify OpenShell's supported host-provider reachability to the loopback runtime. A host alias is not reachability proof; wider binding requires a reviewed network decision. launchd owns process start/stop; this repository adds no launcher, restart daemon or job queue. Stopping the process loses in-flight results.

## Linux image

After the authorized main merge/push and Mac acceptance, build the private runtime from the reviewed commit:

```sh
docker build --platform linux/amd64 --target runtime --build-arg REVISION=<reviewed-commit> -t veoveo-runtime-embedding:0.1.0 .
```

The pinned official Python base plus official `torch==2.14.1+cu130` and `torchvision==0.29.1+cu130` wheels supply userspace CUDA dependencies. Native OpenShell CDI supplies approved NVIDIA devices/driver access. No NVIDIA base image, Docker socket or CPU fallback is added. Runtime uses uid 10001 with an owned `/sandbox` working directory, satisfying native OpenShell workspace write validation. Application code and the virtual environment stay root-owned under `/opt/embedding`, admitted read-only; `PYTHONPATH=/opt/embedding` locates this uninstalled project from the separate workspace, with the absolute Python entrypoint unchanged. Mount the canonical cache read-only at `HF_HOME` and a readable mode-0600 token file. Its `0.0.0.0:8804` listener is for the admitted private container network, never public ingress. Linux/arm64 model startup is unsupported; its base-only check remains supported.

## Private protocol

`GET /livez` is unauthenticated private liveness. `GET /readyz`, `GET /capabilities` and `POST /embeddings` require the fixed bearer. Readiness/capabilities load no model on demand; startup already loaded and verified the fixed GPU model. The MCP client is the public contract/usage boundary.

POST shape: `{requestId: UUIDv7, items: [...]}`. Items use `search_query`/`code_query` with `text`, or `document` with `content` and optional `title`. Unknown members and per-request model settings fail. Limits: 1–8 items, 65,536 total UTF-8 content/title bytes, 8,192 actual tokens/item and 16,384 total tokens including rendered prompts/special tokens. JSON request <=512 KiB; response <=1 MiB. All item/token validation precedes inference. Native `encode(batch_size=2)` bounds padded microbatches while restoring input order.

One admitted request owns the GPU slot, including across microbatches. Busy returns 429 immediately. The 60-second deadline returns ambiguous execution; it cannot cancel an already running GPU kernel. The slot remains occupied until the real worker finishes after timeout/disconnect. Responses are whole batches only. Safe closed problem codes mark explicit `not_started` or `unknown`; arbitrary upstream exception text is never returned. No automatic replay occurs.
