# Runtime ownership

- Read DESIGN.md and README.md. Work only in the assigned worktree; preserve other changes. No subagents or model calls during source iteration. The coordinator owns architecture and rollout; the builder owns full checks, model probes, images and deployment.
- This component owns private authenticated embedding generation through upstream Sentence Transformers/PyTorch. It does not implement MCP, Hub identity, usage accounting, vector storage/search or a custom scheduler.
- Fixed model/revision, prompts, normalization and 768 dimensions; FP32 MPS or CUDA only. No CPU fallback, FP16, per-request models or truncation.
- Use the canonical Hugging Face cache and native pinned revision resolution, with operator prefetch outside serving and read-only runtime access. No second model-manifest or download service.
- Preserve bounded ordered batches, preflight all actual token counts before inference, native encode batch_size=2 and one GPU slot until the actual worker finishes. Caller retry is new computation.
- Never persist/log texts, vectors, tokens or arbitrary upstream errors. Return only closed safe error codes; ambiguous execution must remain explicit.
- Deterministic tests use labelled synthetic fixtures and require no model packages, cache or GPU. Real MPS/CUDA acceptance is a separate measured gate.
- macOS uses native launchd and loopback; Linux uses native OpenShell Docker CUDA/CDI after main merge. No custom supervisor, OpenShell patches or public model endpoint.
- Use uv0.13.1 and the reviewed lock. Base Docker check must work on Linux/arm64 and amd64; inference targets are Mac/arm64 and Linux/amd64.
