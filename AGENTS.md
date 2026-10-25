# Runtime ownership

- Read DESIGN.md and README.md. Work only in the assigned worktree; preserve other changes. No subagents or model calls during source iteration. The coordinator owns architecture and rollout; the builder owns full checks, model probes, images and deployment.
- The design is in [DESIGN.md](DESIGN.md). Change it in the same change as the code.
- This component owns private authenticated embedding generation through upstream Sentence Transformers/PyTorch. It does not implement MCP, Hub identity, usage accounting, vector storage/search or a custom scheduler.
- Fixed full multimodal model/revision; optional request-wide native prompt and 128/256/512/768 dimensions with normalization. FP32 MPS or CUDA inference only; CPU decoding/preprocessing is native, not inference fallback. No per-request model selection.
- Use the canonical Hugging Face cache and native pinned revision resolution, with operator prefetch outside serving and read-only runtime access. No second model-manifest or download service.
- Ordered items contain ordered text/image/audio/video parts, each item yielding one vector. Always pass an explicit outer batch of conversations, including N=1; native encode batch_size=1. Preflight all actual token/media bounds before inference and retain one GPU slot and ephemeral media until the actual worker finishes. Caller retry is new computation.
- MCP alone authorizes Artifact references through the Hub. Runtime consumes private resolved signed GETs from its configured storage origin, verifies bounded bytes and exact size, and uses native loaders. No public URLs, Hub identity, storage credentials, custom sampler, clipping/chunking/aggregation or media persistence. Audio defaults to native truncation=False; explicit truncation is observable.
- Never persist/log texts, vectors, tokens or arbitrary upstream errors. Return only closed safe error codes; ambiguous execution must remain explicit.
- Deterministic tests use labelled synthetic fixtures and require no model packages, cache or GPU. Real MPS/CUDA acceptance is a separate measured gate.
- macOS uses native launchd and loopback; the Linux reference uses OpenShell's native Kubernetes driver and GPU resource/RuntimeClass selection after main merge. Docker remains image-build/component/qualification tooling. No custom supervisor, OpenShell patches or public model endpoint.
- Use uv0.13.1 and the reviewed lock. Base Docker check must work on Linux/arm64 and amd64; inference targets are Mac/arm64 and Linux/amd64.
