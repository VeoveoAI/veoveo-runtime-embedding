# veoveo-runtime-embedding

Private bounded EmbeddingGemma 2 inference behind `veoveo-mcp-embedding`, using upstream Sentence Transformers and PyTorch. This component owns GPU execution; the MCP service owns Hub identity, permissions and usage accounting. Read [DESIGN.md](DESIGN.md) for the runtime contract and the [architecture](https://github.com/VeoveoAI/veoveo-docs/blob/main/ARCHITECTURE.md) for platform responsibilities.

Historical text-only Mac MPS and Linux CUDA deployment qualification passed. The multimodal replacement enables the full model and is being qualified separately; the [execution record](https://github.com/VeoveoAI/veoveo-docs/blob/main/records/OPENSHELL_HARDCUT_PLAN.md#multimodal-embedding-replacement--2026-10-19) distinguishes source checks, native probes and deployed evidence. Synthetic component tests do not prove model inference or media quality.

The fixed full model is `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b`. Ordered text/image/audio/video content produces one normalized FP32 vector per item, jointly embedding all its parts. Dimensions default to 768, with 512/256/128 supported. Targets are native macOS/arm64 MPS and Linux/amd64 CUDA 13 through OpenShell's native Kubernetes GPU resources and RuntimeClass selection. CPU inference, FP16 and user-selected models are unsupported. Media bytes are verified into temporary files owned by the active worker; no durable texts/media/vectors or search API are added.

## Media decoding environment

The `model` extra also installs TorchCodec **0.17.0** on macOS/arm64 and the official **0.17.0+cpu** wheel on Linux/amd64. Hugging Face's native loaders use it for audio/video decoding. They require FFmpeg shared libraries discoverable by the native loader.

The verified Mac installation uses FFmpeg **8.1.1** through `/opt/homebrew/opt/ffmpeg/lib`, resolving to `/opt/homebrew/Cellar/ffmpeg/8.1.1/lib`. The TorchCodec 0.17.0 Mac wheel's FFmpeg 8 library has an `LC_RPATH` of `/opt/homebrew/opt/ffmpeg/lib`; real decoding loaded those libraries with no `DYLD_*` environment settings. Homebrew's versioned `ffmpeg@8` formula is keg-only: installing it alone does not establish this library-discovery arrangement, and that installation was not tested here.

The Docker `model` stage installs Debian trixie's `ffmpeg=7:7.1.5-0+deb13u1`, which supplies FFmpeg 7 shared libraries and the `ffmpeg`/`ffprobe` tools. The base/check stages contain neither FFmpeg nor model dependencies.

Media decoding, audio resampling and video frame sampling are **upstream preprocessing on CPU**. Embedding model inference remains FP32 on explicit MPS/CUDA. The runtime enables all native towers; MCP public API and deployed qualification remain separate integration steps. PyAV, librosa and torchaudio are not needed for the selected native TorchCodec path. See [TorchCodec installation guidance](https://github.com/meta-pytorch/torchcodec/blob/v0.17.0/README.md#installing-torchcodec).

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
| `VEOVEO_EMBEDDING_STORAGE_ORIGIN` | Required exact private object-storage origin with explicit port, e.g. `http://host.openshell.internal:9000`; no path/query/credentials |
| `VEOVEO_EMBEDDING_BIND` | Defaults to `127.0.0.1`; no public listener |
| `VEOVEO_EMBEDDING_PORT` | Defaults to `8804` |

```sh
VEOVEO_EMBEDDING_DEVICE=mps VEOVEO_RUNTIME_REVISION=<reviewed-commit> VEOVEO_EMBEDDING_TOKEN_FILE=<private-file> VEOVEO_EMBEDDING_STORAGE_ORIGIN=http://host.openshell.internal:9000 HF_HOME=<cache-directory> .venv/bin/python -m veoveo_runtime_embedding
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

Both frozen `uv sync` layers use [BuildKit cache mounts](https://docs.astral.sh/uv/guides/integration/docker/#caching), keeping downloaded wheels outside the shipped image while `UV_LINK_MODE=copy` installs its virtual environment.

The pinned official Python base plus official `torch==2.14.1+cu130` and `torchvision==0.29.1+cu130` wheels supply userspace CUDA dependencies. The Linux reference uses OpenShell's native Kubernetes driver with a one-GPU resource request, NVIDIA device plugin and selected `nvidia` RuntimeClass. Docker remains image-build/component/qualification tooling. No NVIDIA base image, Docker socket or CPU fallback is added.

The standalone image retains uid 10001 and its owned, empty mode-0700 `/sandbox` working directory. The reference gateway's selected native service uid/gid is 10001:10001, so its workspace helper can enumerate that image seed as the owner. Use the effective gateway/workspace identity, not the gateway Pod's own user or unused chart values, when preparing mounts. Actual private workspace contents belong to the OpenShell-owned PVC mounted at `/sandbox`, with native ownership; they are not stored in the image seed. Application code and the virtual environment stay root-owned under `/opt/embedding`, admitted read-only; `PYTHONPATH=/opt/embedding` locates this uninstalled project from the separate workspace, with the absolute Python entrypoint unchanged.

Installation owns approved workspace-local cache/token claims and native credential custody. Mount the canonical cache read-only at `HF_HOME` and a readable mode-0600 token file for the actual sandbox identity. The exact Artifact signed-object origin, including explicit port, must match runtime configuration and native endpoint policy; ordinary public TLS trust and end-to-end TLS passthrough are required for this Kubernetes path. Its `0.0.0.0:8804` listener is for the admitted private workload network, never public ingress. Linux/arm64 model startup is unsupported; its base-only check remains supported.

## Private protocol

`GET /livez` is unauthenticated private liveness. `GET /readyz`, `GET /capabilities` and `POST /embeddings` require the fixed bearer. Readiness/capabilities load no model on demand; startup already loaded and verified the fixed GPU model. The MCP client is the public contract/usage boundary.

POST shape: `{requestId: UUIDv7, items: [{content: [...]}], dimensions?, promptName? | prompt?, preprocessing?}`. Every item has ordered parts and produces one vector. Text parts contain `{type:"text", text}`. Private media parts contain `{type:"image" | "audio" | "video", source:{artifactId,mediaType,size,get:{method:"GET",url,headers?,expiresAt}}}`. MCP obtains the descriptor through the caller-bound typed Artifact tool; its public API accepts Artifact IDs instead of descriptors. Signed transfers never appear in embedding output.

The runtime always supplies an explicit outer batch of conversations, including N=1. Native encode uses `batch_size=1` with all token/source preflight completed first. Actual token limits are 8,192/item and 16,384/batch, including prompt/special/media tokens. Unknown members, reserved media placeholders in text/prompts and unsupported native controls fail. JSON request <=512 KiB; response <=1 MiB. Complete ordered outputs require the requested dimension, finite FP32 and unit normalization.

Initial service bounds are 8 items, 16 parts/item, 32 parts/batch, 64 KiB text including prompt, 64 MiB/artifact, 128 MiB total transfers, 16 MP/image or frame, 64 MP aggregate selected pixels, 300 seconds/source audio/video and 600 seconds aggregate decoded audio. FPS is at most 120 and frame budget at most 256. These are **our service limits**, subject to deployed validation, not Google model capabilities. Source metadata is checked before decoding; actual selected shape/sample checks follow.

Only the configured storage origin is admitted. HTTPX uses no environment proxy, no redirects and raw bytes, rejects nonidentity Content-Encoding, and enforces a shared 60-second transfer deadline, signed expiry, exact size and SHA-256. Request/transport loggers are suppressed even with root DEBUG enabled. Both successful and failing transfers avoid signed URLs/headers in logs or errors. The worker owns temporary files through caller timeout/disconnection. Native media validation, token preflight and encode can decode a source **three times**; this bounded cost preserves native path-based behavior without a custom processor.

One admitted request owns the GPU slot, including across microbatches. Busy returns 429 immediately. The 60-second deadline returns ambiguous execution; it cannot cancel an already running GPU kernel. The slot remains occupied until the real worker finishes after timeout/disconnect. Responses are whole batches only. Safe closed problem codes mark explicit `not_started` or `unknown`; arbitrary upstream exception text is never returned. No automatic replay occurs.


## Native options and coverage

`promptName` accepts `SearchQuery`, `CodeRetrieval`, `Document`, `QuestionAnswering`,
`FactChecking`, `Classification`, `Clustering` and `SentenceSimilarity`. Alternatively
pass a literal `prompt`; absent means explicit native empty prompt. Prompts apply once
per invocation, including multimodal items, not per part. Code is ordinary text.

`preprocessing` accepts only these upstream preprocessing controls:

| Group | Controls and defaults |
|---|---|
| `image` | `maxSoftTokens`: 280; choices 70/140/280/560/1120 |
| `audio` | `truncation`: false; when explicitly true, `maxLength` samples, default 480000 |
| `video` | `fps`: 1; `maxFrames`: 32; `overflowStrategy`: uniform or truncate; `maxSoftTokens`: 140; `addTimestamps`: false; `loadAudioFromVideo`: false |

Audio uses native mono 16 kHz decoding/resampling. Truncation defaults off so native
30-second truncation is not silently inherited. Explicit truncation records retained
samples. Native video soundtrack inclusion fails if no audio track exists and cannot
be combined with separate audio parts in this initial contract. Image input uses the
native first frame. Clips, custom longer-media chunking and vector aggregation are
**optional future application preprocessing/postprocessing suggestions**, excluded here.

Every result includes `index`, `vector`, actual `tokenCount` and `media`, an ordered
list of resolved source observations. A media row contains `partIndex`, `type`,
`source:{artifactId,mediaType,size}` and typed `coverage`:

- Image: `width`, `height`, `frameIndex` (zero).
- Audio: `sourceDurationSeconds`, `sourceSamplingRate`, `sourceChannels`,
  `samplingRate` (16000), `decodedSamples`, `retainedSamples`, `truncated`.
- Video: `sourceDurationSeconds`, `sourceFrameCount`, `sourceFps`, `width`, `height`,
  `frameIndices`, `frameTimestampsSeconds` (actual TorchCodec PTS),
  `modelFrameTimestampsSeconds` (native index/FPS prompt values), `audioIncluded`;
  `audio` contains the audio coverage above when included and is otherwise omitted.

Request-level `processing` contains effective `preprocessing`, `promptSha256`, and
optional `promptName`; custom prompt text is not echoed. `processingId` is SHA-256 of
RFC 8785/JCS canonical bytes for the effective fixed model and processing settings,
including the prompt and dimensions. Python uses `rfc8785==0.1.4`; the Rust adapter
independently verifies that identity using its native canonicalization library.
Frozen cross-language vectors live in `tests/fixtures/processing-jcs.json`.
`runtime.ffmpeg` contains `version` plus `libavfilter`, `libavutil`, `libavcodec` and
`libavformat` integer version triples; `runtime.torchcodec` identifies its wheel.

Capabilities distinguish these defaults and service limits from native processing
semantics. Source/token/preprocessing failures are confirmed `not_started`; failures
after native inference starts remain `unknown`. No partial batch is returned.
