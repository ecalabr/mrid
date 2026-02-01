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

