"""Tests for 2.5D slice selection (CNN_NUM_SLICES) and series slice loading."""

import os
import tempfile

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from mrid_app.processing import (get_evenly_spaced_indices, get_sorted_series_files,
                                 load_series_slices)


def test_get_evenly_spaced_indices_spec_example():
    """100-slice series with 3 slices requested -> indices 25, 50, 75."""
    assert get_evenly_spaced_indices(100, 3) == [25, 50, 75]


def test_get_evenly_spaced_indices_center_slice():
    """num_slices=1 must match the legacy center-slice index (len // 2)."""
    for length in [1, 2, 3, 7, 100, 101]:
        assert get_evenly_spaced_indices(length, 1) == [length // 2]


def test_get_evenly_spaced_indices_bounds():
    """Indices stay in range even when num_slices >= series length."""
    for length, num in [(2, 5), (1, 3), (10, 10)]:
        indices = get_evenly_spaced_indices(length, num)
        assert len(indices) == num
        assert all(0 <= i < length for i in indices)


def _write_synthetic_series(out_dir, num_slices=10):
    """Write a synthetic axial DICOM series. Pixel (1, 1) encodes the slice's
    spatial index so tests can confirm which slice was loaded after sorting."""
    series_uid = generate_uid()
    study_uid = generate_uid()
    paths = []
    # Write in shuffled order with names that don't reflect spatial order
    order = list(range(num_slices))
    rng = np.random.RandomState(0)
    rng.shuffle(order)
    for file_idx, slice_idx in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        path = os.path.join(out_dir, f"img_{file_idx:03d}.dcm")
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "MR"
        ds.InstanceNumber = slice_idx + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(slice_idx)]
        ds.Rows = 4
        ds.Columns = 4
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0

        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[0, 1] = 255  # anchor max so normalization preserves values
        arr[1, 1] = slice_idx * 10  # slice marker
        ds.PixelData = arr.tobytes()
        ds.save_as(path)
        paths.append(path)
    return paths


def test_get_sorted_series_files_spatial_order():
    with tempfile.TemporaryDirectory() as d:
        paths = _write_synthetic_series(d, num_slices=10)
        sorted_files = get_sorted_series_files([paths[0]])
        assert len(sorted_files) == 10
        markers = []
        for fpath in sorted_files:
            ds = pydicom.dcmread(fpath, force=True)
            markers.append(int(ds.ImagePositionPatient[2]))
        assert markers == list(range(10))


def test_load_series_slices_center_and_2_5d():
    with tempfile.TemporaryDirectory() as d:
        paths = _write_synthetic_series(d, num_slices=10)

        def marker_of(img):
            return int(round(np.array(img)[1, 1] / 10))

        # N=1 -> center slice (index 5 of 10)
        center = load_series_slices([paths[0]], num_slices=1)
        assert center is not None and len(center) == 1
        assert marker_of(center[0]) == 5

        # N=3 over 10 slices -> indices 2, 5, 7
        slices = load_series_slices([paths[0]], num_slices=3)
        assert slices is not None and len(slices) == 3
        assert [marker_of(img) for img in slices] == [2, 5, 7]


def test_simple_cnn_channel_count():
    try:
        import torch
    except Exception:
        return  # torch not installed; nothing to verify
    from mrid_app.cnn import SimpleCNN, get_configured_num_slices
    from mrid_app.config import IMAGE_SIZE

    assert get_configured_num_slices() >= 1
    model = SimpleCNN(num_classes=4, num_slices=3)
    assert model.conv1.in_channels == 3
    out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (2, 4)


def test_load_cnn_model_infers_slices_from_checkpoint():
    try:
        import torch
    except Exception:
        return  # torch not installed; nothing to verify
    from mrid_app.cnn import SimpleCNN, load_cnn_model

    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "m.pth")
        torch.save(SimpleCNN(num_classes=4, num_slices=3).state_dict(), model_path)
        loaded = load_cnn_model(model_path, num_classes=4)
        assert loaded.conv1.in_channels == 3


def test_preprocessing_cancel_aborts_promptly():
    """prepare_and_cache_cnn_data must return None quickly when cancel_check
    becomes True, without processing the whole dataset."""
    from mrid_app import cnn

    n = 500
    paths = [f"/nonexistent/{i}.dcm" for i in range(n)]
    labels = ["T1"] * n
    calls = {"n": 0}

    def cancel_check():
        calls["n"] += 1
        return calls["n"] >= 2  # cancel almost immediately

    result = cnn.prepare_and_cache_cnn_data(
        paths, labels, lambda p, s: None, "/tmp/mrid_cancel_regression",
        num_workers=2, cancel_check=cancel_check)

    assert result is None
    assert calls["n"] < n  # aborted early, did not run to completion
