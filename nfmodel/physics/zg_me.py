# nfmodel/physics/zg_me.py
import os
import ctypes
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

_lib = None
_cards_dir_cache = None


@contextmanager
def _pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _resolve_lib_path() -> Path:
    env = os.environ.get("ZG_MG5_LIB", "")
    if not env:
        raise FileNotFoundError(
            "ZG_MG5_LIB is not set.\n"
            "Set it to the full path of your MadGraph dylib, e.g.\n"
            "  export ZG_MG5_LIB=/.../libzgme_uux.dylib"
        )
    p = Path(env).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"ZG_MG5_LIB points to a non-existent file: {p}")
    return p


def _resolve_cards_dir() -> Optional[str]:
    """
    Return MG5_CARDS_DIR (where ident_card.dat and param_card.dat live),
    or None if not set.
    Cached after first successful lookup.
    """
    global _cards_dir_cache
    if _cards_dir_cache is not None:
        return _cards_dir_cache

    cards = os.environ.get("MG5_CARDS_DIR", "")
    if not cards:
        return None

    cpath = Path(cards).expanduser()
    if not cpath.exists():
        raise FileNotFoundError(f"MG5_CARDS_DIR does not exist: {cpath}")

    _cards_dir_cache = str(cpath)
    return _cards_dir_cache


def _load():
    global _lib
    if _lib is None:
        lib_path = _resolve_lib_path()
        cards_dir = _resolve_cards_dir()

        # Some MG code may read cards during load/init; so pushd here too.
        if cards_dir is None:
            _lib = ctypes.CDLL(str(lib_path))
        else:
            with _pushd(cards_dir):
                _lib = ctypes.CDLL(str(lib_path))

        _lib.zg_msq.argtypes = [ctypes.POINTER(ctypes.c_double)]
        _lib.zg_msq.restype = ctypes.c_double

    return _lib


def me2(p_all: np.ndarray) -> float:
    """
    p_all shape (4,4): rows [p1, p2, pZ, pg], cols [E,px,py,pz]
    """
    p = np.asarray(p_all, dtype=np.float64).reshape(-1)  # length 16
    lib = _load()

    cards_dir = _resolve_cards_dir()

    # Key fix: lha_read may run during zg_msq evaluation, so pushd per-call.
    if cards_dir is None:
        val = float(lib.zg_msq(p.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
    else:
        with _pushd(cards_dir):
            val = float(lib.zg_msq(p.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))

    if not np.isfinite(val) or val < 0.0:
        raise RuntimeError(
            f"MadGraph returned invalid |M|^2={val}. "
            "Check MG5_CARDS_DIR contains param_card.dat and ident_card.dat."
        )
    return val