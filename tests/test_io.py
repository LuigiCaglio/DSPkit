"""Tests for dspkit.io."""

import io

import numpy as np
import pytest
import scipy.io as sio

from dspkit.io import load_mat

FS = 200.0
N = 1000
rng = np.random.default_rng(0)
X = rng.normal(size=(N, 3))  # MATLAB habit: samples down, channels across


def _mat(**variables) -> bytes:
    buf = io.BytesIO()
    sio.savemat(buf, variables)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------

class TestLayouts:
    def test_matrix_fs_and_labels(self):
        cells = np.array(["a", "b", "c"], dtype=object)
        rec = load_mat(_mat(data=X, fs=FS, labels=cells, t0="2019-01-01T00:00:00"))
        assert rec.data.shape == (3, N)
        np.testing.assert_allclose(rec.data, X.T)
        assert rec.fs == FS
        assert rec.channel_names == ["a", "b", "c"]
        assert rec.t0 == "2019-01-01T00:00:00"
        assert rec.sources == {"data": "data", "fs": "fs",
                               "channel_names": "labels", "t0": "t0"}

    def test_channels_already_in_rows(self):
        rec = load_mat(_mat(y=X.T, Fs=FS))
        assert rec.data.shape == (3, N)
        assert rec.fs == FS

    def test_one_vector_per_channel(self):
        rec = load_mat(_mat(acc1=X[:, 0], acc2=X[:, 1], acc3=X[:, 2],
                            sampling_rate=FS))
        assert rec.data.shape == (3, N)
        assert rec.channel_names == ["acc1", "acc2", "acc3"]
        np.testing.assert_allclose(rec.data[1], X[:, 1])

    def test_time_vector_gives_fs_and_is_not_a_channel(self):
        t = np.arange(N) / FS
        rec = load_mat(_mat(t=t, acc1=X[:, 0], acc2=X[:, 1]))
        assert rec.data.shape == (2, N)
        assert rec.fs == pytest.approx(FS)
        assert rec.sources["fs"] == "t"

    def test_dt(self):
        rec = load_mat(_mat(data=X, dt=1 / FS))
        assert rec.fs == pytest.approx(FS)

    def test_single_struct(self):
        rec = load_mat(_mat(meas={"data": X, "fs": FS}))
        assert rec.data.shape == (3, N)
        assert rec.fs == FS

    def test_char_matrix_names(self):
        names = np.array(["ch1", "ch2", "ch3"])
        rec = load_mat(_mat(data=X, channel_names=names))
        assert rec.channel_names == ["ch1", "ch2", "ch3"]

    def test_no_fs_and_default_names(self):
        rec = load_mat(_mat(data=X))
        assert rec.fs is None
        assert rec.channel_names == ["Signal 1", "Signal 2", "Signal 3"]

    def test_wrong_number_of_labels_is_ignored(self):
        rec = load_mat(_mat(data=X, labels=np.array(["a", "b"], dtype=object)))
        assert rec.channel_names == ["Signal 1", "Signal 2", "Signal 3"]

    def test_reads_from_path(self, tmp_path):
        p = tmp_path / "rec.mat"
        p.write_bytes(_mat(data=X, fs=FS))
        assert load_mat(p).data.shape == (3, N)
        assert load_mat(str(p)).fs == FS


# ---------------------------------------------------------------------------
# explicit variable names
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_named_data_beats_largest(self):
        big = rng.normal(size=(5000, 2))
        rec = load_mat(_mat(raw=big, clean=X, rate=FS), data="clean", fs="rate")
        assert rec.data.shape == (3, N)
        assert rec.fs == FS

    def test_named_time_vector(self):
        rec = load_mat(_mat(data=X, tvec=np.arange(N) / FS), fs="tvec")
        assert rec.fs == pytest.approx(FS)
        assert rec.data.shape == (3, N)

    def test_missing_variable(self):
        with pytest.raises(ValueError, match="No variable 'nope'"):
            load_mat(_mat(data=X), data="nope")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_no_numeric_data(self):
        with pytest.raises(ValueError, match="No numeric array"):
            load_mat(_mat(note="hello", fs=FS))

    def test_v73_is_refused_with_advice(self):
        head = b"MATLAB 7.3 MAT-file".ljust(512, b" ") + b"\x89HDF\r\n\x1a\n" + b"\0" * 64
        with pytest.raises(ValueError, match="-v7"):
            load_mat(head)
