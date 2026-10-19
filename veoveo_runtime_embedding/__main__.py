"""Native process entry; launchd on Mac, native OpenShell Docker on Linux."""
import os
import re
from pathlib import Path
import uvicorn
from .app import create_app
from .backend import ModelBackend


def main():
    # Only trusted startup configuration selects the private endpoint and GPU.
    device = os.environ.get("VEOVEO_EMBEDDING_DEVICE")
    if device not in ("mps", "cuda"):
        raise SystemExit("VEOVEO_EMBEDDING_DEVICE must explicitly name mps or cuda")
    credential_file = os.environ.get("VEOVEO_EMBEDDING_TOKEN_FILE")
    if not credential_file:
        raise SystemExit("VEOVEO_EMBEDDING_TOKEN_FILE is required")
    path = Path(credential_file)
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise SystemExit("runtime token must be a private mode-0600 file")
    credential = path.read_text().strip()
    # The installation records the reviewed source/image identity. No request can select it.
    revision = os.environ.get("VEOVEO_RUNTIME_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("VEOVEO_RUNTIME_REVISION must identify the reviewed 40-character source commit")
    app = create_app(lambda: ModelBackend(device, revision), credential)
    uvicorn.run(app, host=os.environ.get("VEOVEO_EMBEDDING_BIND", "127.0.0.1"),
                port=int(os.environ.get("VEOVEO_EMBEDDING_PORT", "8804")),
                workers=1, access_log=False, log_level="warning", limit_concurrency=16,
                timeout_keep_alive=5, timeout_graceful_shutdown=65)

if __name__ == "__main__":
    main()
