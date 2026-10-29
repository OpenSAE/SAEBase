from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

from src.schema import require_keys


def execute(payload, workdir: Path):
    require_keys(payload, {"files", "corrupt"})
    root = Path(workdir)
    if not isinstance(payload["files"], dict):
        raise ValueError("files")
    for name in payload["files"]:
        if (
            not isinstance(name, str)
            or not name
            or name.startswith(("/", "~"))
            or "\\" in name
            or ":" in name
            or any(part in ("..", ".", "") for part in name.split("/"))
        ):
            raise ValueError("unsafe archive member")
    archive = root / "bundle.tar"
    with tarfile.open(archive, "w") as tar:
        for name, text in sorted(payload["files"].items()):
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    if payload["corrupt"]:
        raise ValueError("simulated archive failure")
    target = root / "extract"
    target.mkdir()
    unpacked = {}
    with tarfile.open(archive, "r") as tar:
        for info in tar.getmembers():
            if info.name not in payload["files"] or not info.isfile():
                raise ValueError("unsafe tar")
            data = tar.extractfile(info).read()
            dest = (target / info.name).resolve()
            if not dest.is_relative_to(target.resolve()):
                raise ValueError("archive escape")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            unpacked[info.name] = hashlib.sha256(data).hexdigest()
    return {
        "files": sorted(unpacked),
        "content_sha256": unpacked,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "verified": True,
    }
