"""Access to OpenADS (https://github.com/farialab/OpenADS).

MedTabX uses the OpenADS lesion-feature code and the JHU/ICBM atlas pack. They
are not part of this package. On first use they are downloaded from GitHub at a
fixed commit into a local cache (about 85 MB; ``git`` is required):

- ``$MEDTABX_OPENADS_DIR`` when it is set, which can also point to an existing
  OpenADS clone,
- otherwise ``~/.cache/medtabx/openads``.

To download in advance, call ``download_openads()``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

OPENADS_REPO = "https://github.com/farialab/OpenADS.git"
OPENADS_COMMIT = "6c378d1ae36278c9351c247e00286f19e0a893ac"
_ATLAS_SUBDIR = Path("assets") / "atlases" / "JHU_ICBM"
_SPARSE_PATHS = ["src", _ATLAS_SUBDIR.as_posix(), "LICENSE", "README.md"]


@lru_cache(maxsize=1)
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_openads_dir() -> Path:
    env = os.environ.get("MEDTABX_OPENADS_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "medtabx" / "openads"


def _is_complete(root: Path) -> bool:
    return ((root / "src" / "ads").is_dir()
            and (root / _ATLAS_SUBDIR / "names.json").is_file())


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def download_openads(target: str | Path | None = None,
                     commit: str = OPENADS_COMMIT) -> Path:
    """Download the OpenADS code and atlas pack at ``commit`` into ``target``."""
    target = Path(target) if target is not None else _default_openads_dir()
    if _is_complete(target):
        return target
    if shutil.which("git") is None:
        raise RuntimeError(
            "OpenADS is not available and git is not installed. Clone "
            f"{OPENADS_REPO} manually and set MEDTABX_OPENADS_DIR to the clone.")
    logger.info("Downloading OpenADS (%s) to %s", commit[:7], target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".openads-", dir=target.parent))
    try:
        _git("init", "-q", cwd=tmp)
        _git("remote", "add", "origin", OPENADS_REPO, cwd=tmp)
        _git("config", "core.sparseCheckout", "true", cwd=tmp)
        (tmp / ".git" / "info" / "sparse-checkout").write_text(
            "\n".join(f"/{p}" for p in _SPARSE_PATHS) + "\n")
        _git("fetch", "-q", "--depth", "1", "--filter=blob:none",
             "origin", commit, cwd=tmp)
        _git("checkout", "-q", "FETCH_HEAD", cwd=tmp)
        if not _is_complete(tmp):
            raise RuntimeError(f"OpenADS commit {commit} lacks the expected files")
        shutil.rmtree(tmp / ".git")
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not download OpenADS from {OPENADS_REPO}: "
            f"{exc.stderr.decode(errors='replace').strip()}") from exc
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return target


@lru_cache(maxsize=1)
def get_openads_root() -> Path:
    root = _default_openads_dir()
    return root if _is_complete(root) else download_openads(root)


def get_openads_src_root() -> Path:
    return get_openads_root() / "src"


def get_openads_atlas_dir() -> Path:
    return get_openads_root() / _ATLAS_SUBDIR


@lru_cache(maxsize=1)
def ensure_openads_on_path() -> Path:
    src_root = get_openads_src_root().resolve()
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return src_root
