"""One-shot uploader for the object-tracking plugin against a FOE deployment.

Usage::

    python _dev_upload.py john-dev
"""

from __future__ import annotations

import sys
from pathlib import Path

# Load the deployment env BEFORE importing fiftyone so FIFTYONE_API_URI /
# FIFTYONE_API_KEY are picked up by the SDK at import time.
sys.path.insert(0, str(Path.home() / "fiftyone-development" / "scripts"))
from _lib import env as _env  # noqa: E402

deployment = sys.argv[1] if len(sys.argv) > 1 else "john-dev"
_env.load(deployment)  # override=True by default

from fiftyone import management as fom  # noqa: E402

PLUGIN_DIR = str(Path(__file__).resolve().parent)
print(f"[upload] deployment={deployment} dir={PLUGIN_DIR}")
result = fom.upload_plugin(PLUGIN_DIR, overwrite=True, optimize=False)
print(f"[upload] uploaded: {result}")
