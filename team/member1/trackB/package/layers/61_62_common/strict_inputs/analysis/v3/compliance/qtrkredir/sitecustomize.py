import os
import sys

_LAYER = os.environ.get("QP_QTRK_LAYER")
if _LAYER is not None:
    import builtins
    import io
    import json
    import re

    _ROOT = os.environ["QP_QTRK_ROOT"].rstrip("/")
    _K = int(_LAYER)
    _QMAP = {"": "qv2sc", "_v1": "qv2sc", "_v2": "qv2", "_v2repro": "qv2", "_v2mo": "qv2mo", "_v2mv": "qv2mv",
             "_v2sc": "qv2sc", "_v2scSIL": "qv2scSIL", "_v2scSILseg2d": "qv2scSILseg2d", "_v3nt": "qv3nt",
             "_v3ntmo": "qv3ntmo", "_v3ntmv": "qv3ntmv", "_v3ntmvSIL": "qv3ntmvSIL",
             "_v3ntmvSILseg2d": "qv3ntmvSILseg2d"}
    _CS = os.environ.get("QP_QTRK_CSUF", "")
    if _CS:
        _QMAP = {k: (v + _CS if v in ("qv2mo", "qv2mv", "qv2sc", "qv2scSIL", "qv3ntmo", "qv3ntmv", "qv3ntmvSIL") else
                     v[:-len("seg2d")] + _CS + "seg2d" if v.endswith("SILseg2d") else v) for k, v in _QMAP.items()}
    _RX = re.compile(r"/geo/out/tracks_(test|val)(_[A-Za-z0-9]+)?/([^/]+)\.json$")
    _GID = {}
    for _sp in ("test",):
        with io.FileIO(f"{_ROOT}/analysis/v3/compliance/qtrack/out/groups_{_sp}.json", "r") as _f:
            _gs = json.loads(_f.read().decode())["groups"]
        _by = {}
        for _g in _gs:
            _by.setdefault(_g["video"], []).append((int(_g["gid"].split("__q")[1]), _g["gid"]))
        for _v, _l in _by.items():
            _l.sort()
            _GID[(_sp, _v)] = _l[_K][1] if _K < len(_l) else _l[0][1]

    def _map(p):
        if isinstance(p, int) or p is None:
            return p
        try:
            s = os.fspath(p)
        except TypeError:
            return p
        isb = isinstance(s, bytes)
        if isb:
            s = os.fsdecode(s)
        a = os.path.abspath(s)
        n = None
        if "/geo/out/tracks_" in a:
            m = _RX.search(a)
            if m and (m.group(2) or "") in _QMAP:
                g = _GID.get((m.group(1), m.group(3)))
                if g is not None:
                    n = f"{_ROOT}/geo/out/tracks_{m.group(1)}_{_QMAP[m.group(2) or '']}/{g}.json"
        if n is None:
            return p
        return os.fsencode(n) if isb else n

    _o_open, _o_osopen = builtins.open, os.open
    _o_exists, _o_isfile, _o_getsize, _o_stat = os.path.exists, os.path.isfile, os.path.getsize, os.stat

    def _open(file, *a, **k):
        return _o_open(_map(file), *a, **k)

    def _osopen(path, *a, **k):
        return _o_osopen(_map(path), *a, **k)

    def _stat(path, *a, **k):
        return _o_stat(_map(path), *a, **k)

    _o_replace, _o_rename, _o_remove, _o_unlink = os.replace, os.rename, os.remove, os.unlink

    def _two(f):
        return lambda src, dst, *a, **k: f(_map(src), _map(dst), *a, **k)

    def _one(f):
        return lambda path, *a, **k: f(_map(path), *a, **k)

    os.replace, os.rename = _two(_o_replace), _two(_o_rename)
    os.remove, os.unlink = _one(_o_remove), _one(_o_unlink)
    builtins.open = _open
    io.open = _open
    os.open = _osopen
    os.stat = _stat
    os.path.exists = lambda p: _o_exists(_map(p))
    os.path.isfile = lambda p: _o_isfile(_map(p))
    os.path.getsize = lambda p: _o_getsize(_map(p))
