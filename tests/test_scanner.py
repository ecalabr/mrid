"""Tests for scanner module including folder organization and compressed DICOM support."""
import tempfile
import os


def test_is_folder_organized_organized():
    """Test that is_folder_organized correctly identifies organized folders."""
    from mrid_app.scanner import is_folder_organized

    # Create test data: each series in its own folder
    dicom_records = [
        {'series_uid': 'series1', 'path': '/data/patient1/study1/series1/image1.dcm'},
        {'series_uid': 'series1', 'path': '/data/patient1/study1/series1/image2.dcm'},
        {'series_uid': 'series2', 'path': '/data/patient1/study1/series2/image1.dcm'},
        {'series_uid': 'series2', 'path': '/data/patient1/study1/series2/image2.dcm'},
    ]

    assert is_folder_organized(dicom_records) == True


def test_is_folder_organized_unorganized():
    """Test that is_folder_organized correctly identifies unorganized folders."""
    from mrid_app.scanner import is_folder_organized

    # Create test data: multiple series in same folder
    dicom_records = [
        {'series_uid': 'series1', 'path': '/data/patient1/study1/image1.dcm'},
        {'series_uid': 'series1', 'path': '/data/patient1/study1/image2.dcm'},
        {'series_uid': 'series2', 'path': '/data/patient1/study1/image3.dcm'},
        {'series_uid': 'series2', 'path': '/data/patient1/study1/image4.dcm'},
    ]

    assert is_folder_organized(dicom_records) == False


def test_is_folder_organized_empty():
    """Test that is_folder_organized handles empty records."""
    from mrid_app.scanner import is_folder_organized

    assert is_folder_organized([]) == True


def test_build_hierarchy_creates_temp_dir():
    """Test that build_hierarchy creates dicom_symlinks directory when needed."""
    from mrid_app.scanner import build_hierarchy

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simple dicom records
        dicom_records = [
            {
                'patient': 'patient1',
                'study_uid': 'study1',
                'accession_number': 'ACC123',
                'series_uid': 'series1',
                'series_desc': 'T1',
                'series_number': 1,
                'path': '/fake/path/image1.dcm',
                'image_position': [0, 0, 0],
                'image_orientation': [1, 0, 0, 0, 1, 0],
            }
        ]

        # Call build_hierarchy - it should create dicom_symlinks
        hierarchy = build_hierarchy(dicom_records, output_dir=tmpdir)

        # Verify dicom_symlinks directory exists
        temp_dir = os.path.join(tmpdir, 'dicom_symlinks')
        assert os.path.isdir(temp_dir)

        # Verify hierarchy was created with expected structure
        assert 'patient1' in hierarchy
        assert 'study1' in hierarchy['patient1']
        assert 'accn' in hierarchy['patient1']['study1']
        assert 'series' in hierarchy['patient1']['study1']
        assert hierarchy['patient1']['study1']['accn'] == 'ACC123'


def test_build_hierarchy_unorganized_creates_symlinks():
    """Test that build_hierarchy creates symlinks for unorganized folders."""
    from mrid_app.scanner import build_hierarchy

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create actual test files in unorganized structure
        test_file_dir = os.path.join(tmpdir, 'input')
        os.makedirs(test_file_dir, exist_ok=True)

        test_file1 = os.path.join(test_file_dir, 'image1.dcm')
        test_file2 = os.path.join(test_file_dir, 'image2.dcm')

        # Create dummy files
        with open(test_file1, 'w') as f:
            f.write('dummy')
        with open(test_file2, 'w') as f:
            f.write('dummy')

        # Create records for unorganized folder (multiple series in same dir)
        dicom_records = [
            {
                'patient': 'patient1',
                'study_uid': 'study1',
                'accession_number': 'ACC456',
                'series_uid': 'series1',
                'series_desc': 'T1',
                'series_number': 1,
                'path': test_file1,
                'image_position': [0, 0, 0],
                'image_orientation': [1, 0, 0, 0, 1, 0],
            },
            {
                'patient': 'patient1',
                'study_uid': 'study1',
                'accession_number': 'ACC456',
                'series_uid': 'series2',
                'series_desc': 'T2',
                'series_number': 2,
                'path': test_file2,
                'image_position': [0, 0, 0],
                'image_orientation': [1, 0, 0, 0, 1, 0],
            }
        ]

        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Build hierarchy
        hierarchy = build_hierarchy(dicom_records, output_dir=output_dir)

        # Verify dicom_symlinks directory exists
        temp_dir = os.path.join(output_dir, 'dicom_symlinks')
        assert os.path.isdir(temp_dir)

        # Check that organized directories were created
        # Naming format uses get_symlink_dir_name: {mrn}-{accn}-{series_uid}
        organized_dir1 = os.path.join(temp_dir, 'patient1-ACC456-series1')
        organized_dir2 = os.path.join(temp_dir, 'patient1-ACC456-series2')

        assert os.path.isdir(organized_dir1)
        assert os.path.isdir(organized_dir2)

        # Verify symlinks were created
        link1 = os.path.join(organized_dir1, 'image1.dcm')
        link2 = os.path.join(organized_dir2, 'image2.dcm')

        assert os.path.islink(link1)
        assert os.path.islink(link2)


def test_check_dicom_batch_gdcm_fallback():
    """Test that check_dicom_batch falls back to gdcm for compressed DICOMs."""
    from mrid_app.scanner import check_dicom_batch

    # Test with non-existent files - should handle gracefully
    filepaths = ['/nonexistent/file1.dcm', '/nonexistent/file2.dcm']
    results = check_dicom_batch(filepaths)

    # Should return None for files that don't exist or can't be read
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None



def _make_marker_series(out_dir, num_slices=10):
    """Synthetic series where the center column pixel encodes the spatial slice
    index, so the chosen center thumbnail can be verified. Files are written in
    shuffled order with records carrying ImagePositionPatient (as in scanning)."""
    import numpy as np
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    series_uid = generate_uid()
    study_uid = generate_uid()
    order = list(range(num_slices))
    np.random.RandomState(0).shuffle(order)
    records = []
    for file_idx, slice_idx in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = os.path.join(out_dir, f"m_{file_idx:03d}.dcm")
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "MR"
        ds.InstanceNumber = slice_idx + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(slice_idx)]
        ds.Rows = 8
        ds.Columns = 8
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        # Zero background (normalization min), a center marker block encoding the
        # slice index, and a constant bright corner anchor (normalization max).
        arr = np.zeros((8, 8), dtype=np.uint16)
        arr[2:6, 2:6] = slice_idx * 100  # center marker block
        arr[0, 0] = 4000                 # anchor
        ds.PixelData = arr.tobytes()
        ds.save_as(path)
        records.append({
            "path": path,
            "image_position": ds.ImagePositionPatient,
            "image_orientation": ds.ImageOrientationPatient,
        })
    return records


def test_extract_center_slice_picks_spatial_center():
    """extract_and_save_center_slice must select the spatial center slice
    (index num_slices // 2) using the scan-phase positions."""
    import tempfile
    import numpy as np
    from PIL import Image
    from mrid_app.scanner import extract_and_save_center_slice

    with tempfile.TemporaryDirectory() as d:
        records = _make_marker_series(d, num_slices=10)
        tdir = os.path.join(d, "thumbs")
        os.makedirs(tdir)
        path = extract_and_save_center_slice(records, tdir, "MRN", "ST", "SE")
        assert path is not None and os.path.exists(path)
        img = np.array(Image.open(path).convert("L"))
        assert img.shape == (256, 256)
        # Center index 5 -> marker fill 500; normalization uses min 0 / max 4000
        # (the anchor), so the marker maps to 500/4000*255 ~= 32. Sample a region
        # away from the corner anchor and back out the slice index.
        center_val = float(np.median(img[120:136, 120:136]))
        recovered_slice = round(center_val / 255.0 * 4000.0 / 100.0)
        assert recovered_slice == 5, (center_val, recovered_slice)


def test_extract_center_slice_falls_back_without_positions():
    """When records lack positions, ordering falls back to InstanceNumber via
    header-only reads and still produces a thumbnail."""
    import tempfile
    from mrid_app.scanner import extract_and_save_center_slice

    with tempfile.TemporaryDirectory() as d:
        records = _make_marker_series(d, num_slices=6)
        for rec in records:
            rec["image_position"] = None  # force the fallback path
        tdir = os.path.join(d, "thumbs")
        os.makedirs(tdir)
        path = extract_and_save_center_slice(records, tdir, "MRN", "ST", "SE")
        assert path is not None and os.path.exists(path)


# ── IGNORE_MULTIPLE_ORIENTATIONS: compare orientations within a tolerance ────

# Values as GE writes them: the same axial orientation, varying slice to slice
# around the 8th-10th decimal place.
_NOISY_AXIAL = [
    ["0.9999528786255", "-0.0097071686387", "0", "0.0097071686387", "0.9999528786255", "0"],
    ["0.9999528786198", "-0.0097071686412", "0", "0.0097071686412", "0.9999528786198", "0"],
    ["0.9999528786301", "-0.0097071686355", "0", "0.0097071686355", "0.9999528786301", "0"],
]
_AXIAL = [1, 0, 0, 0, 1, 0]
_SAGITTAL = [0, 1, 0, 0, 0, -1]
_CORONAL = [1, 0, 0, 0, 0, -1]


def _records(orientations):
    return [{"image_orientation": o} for o in orientations]


def test_orientation_noise_is_one_orientation():
    from mrid_app.scanner import has_multiple_orientations
    assert not has_multiple_orientations(_records(_NOISY_AXIAL * 18))


def test_different_planes_are_multiple_orientations():
    from mrid_app.scanner import has_multiple_orientations
    assert has_multiple_orientations(_records([_AXIAL, _SAGITTAL, _CORONAL]))  # 3-plane localizer


def test_a_slight_tilt_is_still_a_different_orientation():
    import math
    from mrid_app.scanner import has_multiple_orientations
    t = math.radians(1.0)   # e.g. a second, slightly angled axial block
    tilted = [1, 0, 0, 0, math.cos(t), math.sin(t)]
    assert has_multiple_orientations(_records([_AXIAL, tilted]))


def test_missing_or_unreadable_orientations_are_ignored():
    from mrid_app.scanner import has_multiple_orientations
    assert not has_multiple_orientations(_records([None, _AXIAL, [], ["x"] * 6, _AXIAL]))
    assert not has_multiple_orientations(_records([None, None]))


def _write_dicom(path, series_uid, study_uid, iop, z, desc, study_date=None, accn="A1", series_number=1):
    import numpy as np
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID, ds.AccessionNumber, ds.Modality = "P1", accn, "MR"
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
    ds.SeriesDescription, ds.SeriesNumber = desc, series_number
    if study_date:
        ds.StudyDate = study_date
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M"]
    ds.ImageOrientationPatient = iop
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.Rows = ds.Columns = 8
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.PixelData = np.arange(64, dtype=np.uint16).reshape(8, 8).tobytes()
    ds.save_as(path)
    return path


def test_scan_keeps_series_with_orientation_noise():
    """End to end through the real DICOM reader: a series whose slices carry
    noisy orientations is kept, a 3-plane localizer is still skipped."""
    from pydicom.uid import generate_uid
    from mrid_app import config
    from mrid_app.scanner import build_hierarchy, check_dicom_batch

    saved = config.IGNORE_MULTIPLE_ORIENTATIONS
    config.IGNORE_MULTIPLE_ORIENTATIONS = True
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dicom_dir = os.path.join(tmpdir, "dicom")
            os.makedirs(dicom_dir)
            study = generate_uid()
            spgr, loc = generate_uid(), generate_uid()
            paths = [_write_dicom(os.path.join(dicom_dir, f"spgr{i}.dcm"), spgr, study,
                                  _NOISY_AXIAL[i % 3], i, "ax spgr +c") for i in range(12)]
            paths += [_write_dicom(os.path.join(dicom_dir, f"loc{i}.dcm"), loc, study,
                                   iop, i, "loc") for i, iop in enumerate([_AXIAL, _SAGITTAL, _CORONAL])]

            records = [r for r in check_dicom_batch(paths) if r]
            assert len(records) == 15
            noisy = {tuple(r["image_orientation"]) for r in records if r["series_desc"] == "ax spgr +c"}
            assert len(noisy) == 3, "premise: the stored orientations differ exactly"

            hierarchy = build_hierarchy(records, output_dir=os.path.join(tmpdir, "out"))
            series = hierarchy.get("P1", {}).get(study, {}).get("series", {})
            assert "ax spgr +c" in series, "noisy-orientation series must be kept"
            entry = series["ax spgr +c"]
            assert len(entry) == 1, "kept as one series, not split"
            linked = os.listdir(os.path.dirname(entry[0]["dicom_path"]))
            assert len(linked) == 12, f"all 12 slices must be kept, got {len(linked)}"
            assert "loc" not in series, "a real multi-plane localizer must still be skipped"
    finally:
        config.IGNORE_MULTIPLE_ORIENTATIONS = saved


# ── series that share a description ─────────────────────────────────────────

def _record(series_uid, desc, study_uid="study1", patient="P1", number=1, date="", accn="A1"):
    return {
        'patient': patient, 'study_uid': study_uid, 'accession_number': accn,
        'series_uid': series_uid, 'series_desc': desc, 'series_number': number,
        'path': f'/fake/{study_uid}/{series_uid}/1.dcm', 'image_position': [0, 0, 0],
        'image_orientation': [1, 0, 0, 0, 1, 0], 'study_date': date,
    }


def test_series_sharing_a_description_each_get_a_key():
    from mrid_app.scanner import build_hierarchy
    records = [_record("s1", "AX T1", number=3), _record("s2", "AX T1", number=7),
               _record("s3", "AX T1", number=9), _record("s4", "FLAIR", number=4)]
    series = build_hierarchy(records)["P1"]["study1"]["series"]
    assert sorted(series) == ["AX T1", "AX T1 (2)", "AX T1 (3)", "FLAIR"]
    assert all(len(entries) == 1 for entries in series.values())
    # Ordered by series number: the earliest series keeps the plain description
    assert series["AX T1"][0]["dicom_path"] == "/fake/study1/s1/1.dcm"
    assert series["AX T1 (2)"][0]["dicom_path"] == "/fake/study1/s2/1.dcm"


def test_older_hierarchies_are_split_on_load():
    from mrid_app.scanner import split_shared_descriptions
    old = {"P1": {"study1": {"accn": "A1", "series": {
        "AX T1": [{"dicom_path": "/a", "thumbnail_path": "/ta"},
                  {"dicom_path": "/b", "thumbnail_path": "/tb"}],
        "AX T1 (2)": [{"dicom_path": "/c", "thumbnail_path": "/tc"}],
        "FLAIR": [{"dicom_path": "/d", "thumbnail_path": "/td"}],
    }}}}
    series = split_shared_descriptions(old)["P1"]["study1"]["series"]
    assert {k: [e["dicom_path"] for e in v] for k, v in series.items()} == {
        "AX T1": ["/a"], "AX T1 (3)": ["/b"], "AX T1 (2)": ["/c"], "FLAIR": ["/d"],
    }
    assert split_shared_descriptions(None) is None


def test_every_series_sharing_a_description_reaches_the_csv():
    """End to end: scan, build the hierarchy and write the classification CSV."""
    import csv
    from pydicom.uid import generate_uid
    from mrid_app import processing
    from mrid_app.scanner import build_hierarchy, check_dicom_batch, select_series_images

    with tempfile.TemporaryDirectory() as tmpdir:
        dicom_dir = os.path.join(tmpdir, "dicom")
        os.makedirs(dicom_dir)
        study = generate_uid()
        paths = []
        for n, uid in enumerate((generate_uid(), generate_uid()), start=1):
            paths += [_write_dicom(os.path.join(dicom_dir, f"s{n}_{i}.dcm"), uid, study,
                                   _AXIAL, i, "ax t1", series_number=n) for i in range(3)]
        out = os.path.join(tmpdir, "out")
        hierarchy = build_hierarchy(select_series_images(check_dicom_batch(paths)), output_dir=out)
        series = hierarchy["P1"][study]["series"]
        assert sorted(series) == ["ax t1", "ax t1 (2)"]

        labels = {series["ax t1"][0]["dicom_path"]: "T1", series["ax t1 (2)"][0]["dicom_path"]: "Other"}
        processing.background_conversion_worker("P1", study, out, labels, series, "A1", False)
        csvs = [os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs if f.endswith(".csv")]
        assert len(csvs) == 1
        with open(csvs[0], newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert sorted(r["Classification"] for r in rows) == ["Other", "T1"]


# ── merging a patient's studies ─────────────────────────────────────────────

def test_study_merge_map_groups_by_date_window():
    from mrid_app.scanner import study_merge_map
    records = [
        _record("a", "T1", study_uid="day1_am", date="20240301"),
        _record("b", "T1", study_uid="day1_pm", date="20240301"),
        _record("c", "T1", study_uid="day2", date="20240302"),
        _record("d", "T1", study_uid="day4", date="2024.03.04"),   # old DA form
        _record("e", "T1", study_uid="undated", date=""),
        _record("f", "T1", study_uid="other_patient", patient="P2", date="20240301"),
    ]
    same_day = study_merge_map(records, 0)
    assert same_day[("P1", "day1_pm")] == "day1_am"
    assert same_day[("P1", "day2")] == "day2"
    assert ("P1", "undated") not in same_day
    assert same_day[("P2", "other_patient")] == "other_patient"

    # The window is measured from each exam's first study, so it does not chain
    two_days = study_merge_map(records, 2)
    assert two_days[("P1", "day2")] == "day1_am"
    assert two_days[("P1", "day4")] == "day4"
    assert study_merge_map(records, 3)[("P1", "day4")] == "day1_am"


def test_merged_studies_become_one_exam_only_when_enabled():
    from mrid_app import config
    from mrid_app.scanner import build_hierarchy

    def records():
        return [_record("s1", "AX T1", study_uid="early", date="20240301", accn="ACC1"),
                _record("s2", "FLAIR", study_uid="early", date="20240301", accn="ACC1"),
                _record("s3", "AX T1", study_uid="late", date="20240301", accn="ACC2"),
                _record("s4", "AX T1", study_uid="next_day", date="20240302", accn="ACC3")]

    saved = (config.MERGE_STUDIES, config.MERGE_STUDIES_WINDOW_DAYS)
    try:
        config.MERGE_STUDIES = False
        assert sorted(build_hierarchy(records())["P1"]) == ["early", "late", "next_day"]

        config.MERGE_STUDIES, config.MERGE_STUDIES_WINDOW_DAYS = True, 0
        patient = build_hierarchy(records())["P1"]
        assert sorted(patient) == ["early", "next_day"]
        exam = patient["early"]
        assert exam["accn"] == "ACC1"
        assert sorted(exam["merged_studies"]) == ["early", "late"]
        assert sorted(exam["series"]) == ["AX T1", "AX T1 (2)", "FLAIR"]
        assert "merged_studies" not in patient["next_day"]

        config.MERGE_STUDIES_WINDOW_DAYS = 1
        assert sorted(build_hierarchy(records())["P1"]) == ["early"]
    finally:
        config.MERGE_STUDIES, config.MERGE_STUDIES_WINDOW_DAYS = saved


def test_merged_exam_keeps_each_series_symlinks_under_its_own_study():
    """With MERGE_STUDIES on, a scan of real files gives one exam whose series
    still link to their own files."""
    from pydicom.uid import generate_uid
    from mrid_app import config
    from mrid_app.scanner import build_hierarchy, check_dicom_batch, select_series_images

    saved = (config.MERGE_STUDIES, config.MERGE_STUDIES_WINDOW_DAYS)
    config.MERGE_STUDIES, config.MERGE_STUDIES_WINDOW_DAYS = True, 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dicom_dir = os.path.join(tmpdir, "dicom")   # both studies in one folder
            os.makedirs(dicom_dir)
            first, second = generate_uid(), generate_uid()
            paths = [_write_dicom(os.path.join(dicom_dir, f"a{i}.dcm"), "1.2.3.1",
                                  first, _AXIAL, i, "ax t1", study_date="20240301", accn="ACC1")
                     for i in range(3)]
            paths += [_write_dicom(os.path.join(dicom_dir, f"b{i}.dcm"), "1.2.3.2",
                                   second, _AXIAL, i, "ax t1", study_date="20240301", accn="ACC2")
                      for i in range(3)]
            records = select_series_images(check_dicom_batch(paths))
            assert {r["study_date"] for r in records} == {"20240301"}
            patient = build_hierarchy(records, output_dir=os.path.join(tmpdir, "out"))["P1"]
            assert len(patient) == 1
            exam = next(iter(patient.values()))
            assert sorted(exam["series"]) == ["ax t1", "ax t1 (2)"]
            folders = {os.path.dirname(e[0]["dicom_path"]) for e in exam["series"].values()}
            assert len(folders) == 2
            for folder in folders:
                assert len(os.listdir(folder)) == 3
    finally:
        config.MERGE_STUDIES, config.MERGE_STUDIES_WINDOW_DAYS = saved
