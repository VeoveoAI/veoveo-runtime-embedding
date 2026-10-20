# Private multimodal embedding runtime

This component runs the fixed `google/embeddinggemma-2` revision
`914f7f89142e33e77833254d9c9b90c3cef7303b` through pinned Sentence Transformers,
Transformers, PyTorch and TorchCodec. The full vision/audio/language encoders remain
enabled. Model parameters and inference are FP32 on explicit MPS or CUDA; native CPU
media decoding and preprocessing are separate operations. No CPU inference fallback,
per-request model selection, custom scheduler or encoder patches exist.

The Rust Embedding MCP owns the public contract, Hub permissions, caller-bound typed
Artifact authorization and Core usage. Runtime has no Hub identity or Artifact ACL
implementation. A fixed private bearer admits resolved requests. It stores no durable
content or vectors and performs no search.

## Ordered model input

A request has ordered `items`; each has ordered `content` parts. Text parts contain
`type` and `text`. Private media parts contain `type` and a closed `source` descriptor
with Artifact identity, media type, size and Artifact's signed GET. Public
media references remain Artifact IDs; callers cannot supply a runtime URL. ArtifactID
and create-only storage identify the bytes; the runtime does not inherit an Artifact
content-digest requirement. Exact-size, bounded transfer and native decoding admission
remain mandatory.

Each item becomes one native structured conversation with ordered content. The batch
is **always** an explicit outer list of conversations, including a single item. One
item produces one joint vector, not independent vectors averaged by the service.
Text/code share the text input type. Native `encode(batch_size=1)` restores order;
complete output cardinality, requested dimension, FP32, finiteness and unit norms are
validated before returning anything.

The default prompt is explicitly empty. A request may select one of eight pinned
native prompt names or a literal prompt, exclusively. The same literal prompt is
passed once to native preprocessing and encode. Media placeholders in caller text or
prompts are refused: the pinned chat template treats those specially and would
otherwise suppress automatic ordered media markers. Fresh conversation copies are
required for each call because native audio-from-video processing appends a marker
in place. No per-part prompt setting is invented.

Dimensions are 768 by default, with native Matryoshka outputs of 512/256/128 exposed
through Sentence Transformers truncation followed by normalization. The processing
identity hashes RFC 8785/JCS canonical bytes of the fixed checkpoint, effective dimensions/prompt/preprocessing,
ordered layout, pooling and audio rate. It excludes input content other than an
explicit prompt. Execution versions/devices remain separate provenance. Custom
prompt text is not echoed in output; output includes its SHA-256.
Python uses pinned `rfc8785==0.1.4`; the Rust MCP independently checks the same JCS
SHA-256. Frozen cross-language vectors cover UTF-8 prompts, integral FPS represented
as either integer or float, and small fractional FPS. Ordinary JSON serialization is
not the processing identity format.

## Media transfer and native preprocessing

`VEOVEO_EMBEDDING_STORAGE_ORIGIN` is required trusted startup configuration: one
HTTP(S) origin with explicit port and no credentials/path/query/fragment. Private HTTP
is allowed only at that exact configured origin. HTTPX 0.28.1 streams raw bytes with
proxy environment disabled, no redirects and a 60-second total deadline across all
request transfers. Status must be 200; nonidentity content encoding, mismatched
Content-Length, actual byte count or SHA-256, expired grants and foreign Host headers
fail before model inference. Every signed header is otherwise preserved. HTTPX and
HTTPCore transport loggers and descendants are suppressed independently of root log
level. Transfer descriptors hide URLs/headers in their representation; closed errors
never include upstream exceptions, source URLs or content.

Verified files belong to a worker-owned TemporaryDirectory. TorchCodec metadata and
PIL headers admit all source dimensions/durations before any pixel/sample validation
decode. Video indices come from the pinned processor's actual `sample_frames` with
the effective native options. Native TorchCodec then validates selected decoded
shapes, records actual frame PTS and decoded mono 16 kHz sample counts, and releases
its decoder buffers. Source metadata is an admission check, not a guarantee against
all codec/parser resource use on malformed files; source byte bounds and final actual
shape/sample checks also apply.
After each validation decode, actual audio samples from explicit audio or video
soundtracks also count against the aggregate 600-second bound. An underreported
source duration therefore cannot bypass it; overflow stops subsequent validation
decodes and prevents token preprocessing or model inference.

The model receives verified **local paths**, never a signed URL. The validation
decode, native token-preflight decode and native encode decode can therefore each
read the bounded media: **up to three decodes**. This explicit cost preserves native
video/audio behavior and avoids a custom media processor or patch. Native video
prompt timestamps use index/source FPS; coverage reports those separately from actual
decoded PTS. An image input represents its native first frame, reported as frame zero.

Video defaults are 1 FPS, 32 frames, uniform overflow, 140 soft tokens/frame, no
prompt timestamps and no soundtrack. `loadAudioFromVideo` uses native soundtrack
loading/placeholder insertion. A missing soundtrack fails explicitly, and combining
this flag with separate audio parts is refused because of the pinned upstream branch
interaction. Audio defaults to `truncation=False`, preserving the native >30-second
path. Explicit `truncation=True` permits `maxLength` samples (native default 480000
if omitted); coverage reports decoded/retained counts and whether truncation occurred.
These are upstream preprocessing controls. Clips, custom long-media chunking and
embedding aggregation are optional future application pre/postprocessing proposals,
excluded from this implementation.

## Service admission and execution boundary

The following are Veoveo service admission bounds, not Google model capabilities or
quality guarantees: 8 items; 16 parts/item and 32 parts/batch; 64 KiB text including
prompt; 64 MiB/artifact and 128 MiB total transfers; 16 megapixels/image or frame and
64 megapixels total selected images/frames; 300 seconds/source audio or video and
600 seconds aggregate decoded audio; FPS at most 120 and frame budget at most 256.
These defensive initial limits require deployed capacity qualification.

Native untruncated preprocessing includes special tokens and multimodal placeholders.
Each item must have exactly one preflight row, at most 8192 actual tokens; the batch
must total at most 16384. Observed image/audio/video tensor cardinalities and sampled
frame counts must match intended coverage. All source/token checks precede the first
model forward. Native max-length metadata is not rewritten to implement these bounds.

Source/preprocessing failures are closed `Rejected` errors with confirmed
`not_started` execution. Once native inference is invoked, failures and invalid
outputs remain ambiguous. The MCP preserves conservative estimates for ambiguous
outcomes. HTTP timeout/cancellation shields the actual worker; the one GPU slot and
TemporaryDirectory remain owned until that worker finishes. A retry is new work.
There is no queue, replay, GPU-kernel cancellation promise or durable Task here.

## Environment and evidence

Startup resolves the exact canonical Hugging Face cache offline, checks pinned package
identities, full towers, prompt/pooling contract, GPU availability and every parameter's
FP32 placement. Runtime reports TorchCodec and actual linked FFmpeg library versions.
No forward hooks or wrappers are installed in serving; actual GPU-forward observations
belong to separate qualification probes.

TorchCodec 0.17.0 runs on Mac arm64; Linux AMD64 uses the official 0.17.0+cpu decoder
with CUDA 13 inference wheels. FFmpeg shared libraries are required. Docker's model
stage installs Debian FFmpeg 7.1.5; base/check stages remain model-free on AMD64 and
ARM64. Mac's verified installation and library discovery are documented in README.

Deterministic tests use labeled synthetic counts/vectors/decoder seams and real
loopback HTTP transfer fixtures. They prove contract, admission, integrity, timeout,
cleanup and delegation-free runtime boundaries; they do not prove model inference,
media semantic quality or encoding validity of their synthetic transfer bytes. Full
model MPS/CUDA and deployed Hub qualification are distinct measured gates. Historical
text-only deployment evidence does not qualify this replacement.
