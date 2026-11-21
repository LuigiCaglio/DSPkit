"""
Reading measurement files.

Functions
---------
load_mat            -- read a MATLAB .mat file into channels, sample rate and names

Classes
-------
MatRecord           -- what load_mat returns
"""

from __future__ import annotations

import io as _io
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import scipy.io as _sio


# Variable names recognised for each role. Compared after lower-casing and
# dropping everything but letters and digits, so `Fs`, `fs_Hz` and
# `sampling_rate` all land on the same key.
_FS_NAMES = {"fs", "fshz", "fsamp", "fsample", "samplerate", "samplingrate",
             "samplingfrequency", "samplefrequency", "srate", "sr"}
_DT_NAMES = {"dt", "ts", "samplinginterval", "sampleinterval", "timestep"}
_TIME_NAMES = {"t", "time", "times", "tt", "timevector", "timeaxis"}
_NAME_NAMES = {"labels", "label", "names", "channelnames", "channels",
               "channellabels", "chnames", "channames", "sensornames", "sensors"}
_T0_NAMES = {"t0", "starttime", "start", "tstart", "timestamp", "datetime"}

PathOrFile = Union[str, "os.PathLike[str]", bytes, _io.IOBase]


@dataclass
class MatRecord:
    """
    A multi-channel record read from a .mat file.

    Attributes
    ----------
    data : ndarray, shape (n_channels, n_samples)
        One row per channel.
    fs : float or None
        Sampling frequency in Hz, or None if the file does not say.
    channel_names : list of str
        One name per row of ``data``.
    t0 : str or None
        Start time of the record, if the file carries one, as written in it.
    sources : dict
        Which variable in the file supplied each of ``data``, ``fs``,
        ``channel_names`` and ``t0`` -- so a guess can be checked rather than
        trusted.
    """

    data: np.ndarray
    fs: Optional[float]
    channel_names: list
    t0: Optional[str] = None
    sources: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _is_v73(head: bytes) -> bool:
    return head[:19] == b"MATLAB 7.3 MAT-file" or head[512:520] == b"\x89HDF\r\n\x1a\n"


def _read_variables(source: PathOrFile) -> dict:
    """The file's variables, with MATLAB structs as SimpleNamespace-like objects."""
    if isinstance(source, (bytes, bytearray)):
        source = _io.BytesIO(source)

    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as fh:
            head = fh.read(528)
    else:
        pos = source.tell()
        head = source.read(528)
        source.seek(pos)

    if _is_v73(head):
        raise ValueError(
            "This is a MATLAB v7.3 (HDF5) .mat file, which is not supported yet. "
            "Re-save it in MATLAB with save(filename, ..., '-v7')."
        )

    raw = _sio.loadmat(source, squeeze_me=True, struct_as_record=False,
                       chars_as_strings=True)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def _is_struct(v) -> bool:
    return hasattr(v, "_fieldnames")


def _struct_fields(v) -> dict:
    return {f: getattr(v, f) for f in v._fieldnames}


def _as_scalar(v) -> Optional[float]:
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.size != 1:
        return None
    x = float(a.ravel()[0])
    return x if np.isfinite(x) else None


def _as_strings(v) -> Optional[list]:
    """A cell array of strings or a char matrix, as a list of str."""
    if isinstance(v, str):
        return [v]
    a = np.asarray(v)
    if a.dtype.kind in "US":
        return [str(s).strip() for s in a.ravel()]
    if a.dtype == object and a.size and all(isinstance(s, str) for s in a.ravel()):
        return [s.strip() for s in a.ravel()]
    return None


def _is_numeric_array(v) -> bool:
    if isinstance(v, np.ndarray):
        return v.dtype.kind in "fiub" and v.ndim in (1, 2) and v.size > 1
    return False


def _find(variables: dict, names: set, test) -> "tuple[Optional[str], object]":
    for k, v in variables.items():
        if _key(k) in names:
            got = test(v)
            if got is not None:
                return k, got
    return None, None


def _uniform_fs(t: np.ndarray) -> Optional[float]:
    d = np.diff(np.asarray(t, dtype=float))
    if d.size < 2 or not np.all(d > 0):
        return None
    med = float(np.median(d))
    if np.std(d) / med > 1e-3:
        return None
    return 1.0 / float(np.mean(d))


def load_mat(
    source: PathOrFile,
    data: Optional[str] = None,
    fs: Optional[str] = None,
    names: Optional[str] = None,
) -> MatRecord:
    """
    Read a MATLAB .mat file as channels, a sample rate and channel names.

    No particular layout is required. Each role is looked for by variable name
    (case and punctuation ignored), and every choice made is reported in
    ``MatRecord.sources``:

    * **data** -- the largest numeric 1-D or 2-D array. Signals are taken to run
      along the long axis, since a record has far more samples than channels.
      If the file instead holds one vector per channel (``acc1``, ``acc2``,
      ...), every vector of the most common length is stacked, and the
      variable names become the channel names.
    * **sample rate** -- a scalar named ``fs``, ``Fs``, ``sampling_rate``,
      ``srate``, ...; else a sample interval named ``dt``, ``Ts``, ...; else a
      uniformly spaced time vector named ``t`` or ``time``. A time vector is
      never returned as a channel.
    * **channel names** -- a cell array or char matrix named ``labels``,
      ``names``, ``channels``, ``channel_names``, ... with one entry per
      channel.
    * **start time** -- a string named ``t0``, ``start_time``, ``timestamp``, ...

    A file whose only variable is a struct (``s.data``, ``s.fs``) is read
    from inside that struct.

    Parameters
    ----------
    source : str, path-like, bytes or binary file object
        The .mat file. MATLAB v4, v6 and v7 files are supported; v7.3 (HDF5)
        files are not, and raise with a note on how to re-save them.
    data, fs, names : str, optional
        Variable names to use instead of searching. ``fs`` may name either a
        sample rate or a time vector.

    Returns
    -------
    MatRecord
        ``data`` has shape (n_channels, n_samples). ``fs`` is None when the file
        gives no way to tell.

    Raises
    ------
    ValueError
        If the file holds no numeric array that could be a signal, if a named
        variable is missing, or if the file is v7.3.

    Examples
    --------
    >>> rec = dsp.load_mat("ambient_20190101_000000.mat")
    >>> rec.data.shape, rec.fs, rec.channel_names[:2]
    ((6, 247742), 825.806..., ['aBD11Az', 'aBD17Ay'])
    """
    variables = _read_variables(source)

    # A single struct holding everything: look inside it instead.
    if len(variables) == 1:
        (only,) = variables.values()
        if _is_struct(only):
            variables = _struct_fields(only)

    sources: dict = {}

    def named(var: Optional[str], role: str):
        if var is None:
            return None
        if var not in variables:
            raise ValueError(
                f"No variable '{var}' in this file for {role}. "
                f"It has: {', '.join(sorted(variables)) or 'nothing'}."
            )
        return variables[var]

    # ── time vector (identified first, so it is not mistaken for a channel) ──
    time_var: Optional[str] = None
    if fs is not None and _as_scalar(named(fs, "the sample rate")) is None:
        time_var = fs
    else:
        for k, v in variables.items():
            if _key(k) in _TIME_NAMES and _is_numeric_array(v) and np.asarray(v).ndim == 1:
                time_var = k
                break

    # ── data ────────────────────────────────────────────────────────────────
    if data is not None:
        arr = np.asarray(named(data, "the data"))
        if arr.dtype.kind not in "fiub" or arr.ndim not in (1, 2):
            raise ValueError(f"'{data}' is not a 1-D or 2-D numeric array.")
        data_vars = [data]
    else:
        candidates = {
            k: v for k, v in variables.items()
            if _is_numeric_array(v) and k != time_var
        }
        if not candidates:
            raise ValueError(
                "No numeric array in this .mat file looks like signal data. "
                f"Variables found: {', '.join(sorted(variables)) or 'none'}."
            )
        largest = max(candidates, key=lambda k: candidates[k].size)
        if candidates[largest].ndim == 2 and min(candidates[largest].shape) > 1:
            arr = candidates[largest]
            data_vars = [largest]
        else:
            # One vector per channel. Stack every vector sharing the most common
            # length, in file order, so a stray short vector is left out.
            vectors = {k: np.ravel(v) for k, v in candidates.items()}
            lengths = [v.size for v in vectors.values()]
            common = max(set(lengths), key=lambda n: (lengths.count(n), n))
            data_vars = [k for k, v in vectors.items() if v.size == common]
            arr = np.vstack([vectors[k] for k in data_vars])

    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if len(data_vars) == 1 and arr.shape[0] > arr.shape[1]:
        arr = arr.T  # signals run along the long axis
    n_ch, n_samp = arr.shape
    sources["data"] = data_vars[0] if len(data_vars) == 1 else data_vars

    # ── sample rate ─────────────────────────────────────────────────────────
    rate: Optional[float] = None
    if fs is not None and time_var != fs:
        rate = _as_scalar(variables[fs])
        sources["fs"] = fs
    else:
        k, val = _find(variables, _FS_NAMES, _as_scalar)
        if val is not None and val > 0:
            rate, sources["fs"] = val, k
        else:
            k, val = _find(variables, _DT_NAMES, _as_scalar)
            if val is not None and val > 0:
                rate, sources["fs"] = 1.0 / val, k
            elif time_var is not None:
                t = np.ravel(variables[time_var])
                if t.size == n_samp:
                    rate = _uniform_fs(t)
                    if rate is not None:
                        sources["fs"] = time_var

    # ── channel names ───────────────────────────────────────────────────────
    labels: Optional[list] = None
    if names is not None:
        labels = _as_strings(named(names, "the channel names"))
        if labels is None:
            raise ValueError(f"'{names}' is not a list of strings.")
        sources["channel_names"] = names
    else:
        for k, v in variables.items():
            if _key(k) in _NAME_NAMES:
                got = _as_strings(v)
                if got is not None and len(got) == n_ch:
                    labels, sources["channel_names"] = got, k
                    break
    if labels is None and len(data_vars) > 1:
        labels = list(data_vars)
    if labels is None or len(labels) != n_ch:
        labels = [f"Signal {i + 1}" for i in range(n_ch)]

    # ── start time ──────────────────────────────────────────────────────────
    t0: Optional[str] = None
    for k, v in variables.items():
        if _key(k) in _T0_NAMES:
            got = _as_strings(v)
            if got and len(got) == 1 and got[0]:
                t0, sources["t0"] = got[0], k
                break

    return MatRecord(data=arr, fs=rate, channel_names=labels, t0=t0, sources=sources)
