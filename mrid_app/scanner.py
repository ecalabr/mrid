from collections import defaultdict


def check_dicom(filepath, include_types=None, exclude_types=None):
    """Worker function executed in parallel.

    Returns minimal DICOM metadata or None. Filters to only ORIGINAL + PRIMARY images
    by default. This implementation minimizes disk I/O by performing a small
    preamble check and requesting only the tags we need when pydicom supports it.
    """
    try:
        # Quick preamble check (cheap). Most DICOMs have 'DICM' at offset 128.
        try:
            with open(filepath, 'rb') as fh:
                fh.seek(128)
                pre = fh.read(4)
                has_preamble = (pre == b'DICM')
        except Exception:
            has_preamble = False

        import pydicom
        ds = None
        # Prepare a set of tags we need; pydicom may accept specific_tags kwarg
        try:
            from pydicom.tag import Tag
            specific = [
                Tag(0x0008, 0x0008),  # ImageType
                Tag(0x0010, 0x0020),  # PatientID
                Tag(0x0020, 0x000D),  # StudyInstanceUID
                Tag(0x0020, 0x000E),  # SeriesInstanceUID
                Tag(0x0008, 0x103E),  # SeriesDescription
                Tag(0x0020, 0x0011),  # SeriesNumber
            ]
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True, specific_tags=specific)
            except TypeError:
                # older pydicom doesn't support specific_tags
                ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
        except Exception:
            # If anything goes wrong with Tag or specific_tags, fall back to full header read
            ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)

        img_type = getattr(ds, "ImageType", [])
        include_types = include_types or ["ORIGINAL", "PRIMARY"]
        exclude_types = exclude_types or ["DERIVED", "SECONDARY", "ADC", "DIFFUSION", "LOCALIZER", "MIP", "MPR"]

        # simple checks
        if not all([item in img_type for item in include_types]):
            return None
        if any([item in img_type for item in exclude_types]):
            return None

        return {
            "path": filepath,
            "patient": getattr(ds, "PatientID", "UNKNOWN_MRN"),
            "study_uid": str(getattr(ds, "StudyInstanceUID", None)),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", None)),
            "series_desc": getattr(ds, "SeriesDescription", "UNKNOWN_SERIES"),
            "series_number": int(getattr(ds, "SeriesNumber", 0)),
        }
    except Exception:
        return None


def check_dicom_batch(filepaths, include_types=None, exclude_types=None):
    """Process a batch of file paths and return a list of dicom metadata records (or None entries).

    This reduces per-file overhead by importing and initializing pydicom once per worker
    process and then parsing many files sequentially.
    """
    results = []
    try:
        import pydicom
        try:
            from pydicom.tag import Tag
            have_tag = True
        except Exception:
            have_tag = False

        include_types = include_types or ["ORIGINAL", "PRIMARY"]
        exclude_types = exclude_types or ["DERIVED", "SECONDARY", "ADC", "DIFFUSION", "LOCALIZER", "MIP", "MPR"]

        # Precreate specific tags if supported
        specific = None
        if have_tag:
            try:
                specific = [
                    Tag(0x0008, 0x0008),  # ImageType
                    Tag(0x0010, 0x0020),  # PatientID
                    Tag(0x0020, 0x000D),  # StudyInstanceUID
                    Tag(0x0020, 0x000E),  # SeriesInstanceUID
                    Tag(0x0008, 0x103E),  # SeriesDescription
                    Tag(0x0020, 0x0011),  # SeriesNumber
                ]
            except Exception:
                specific = None

        for fp in filepaths:
            try:
                # cheap preamble check
                try:
                    with open(fp, 'rb') as fh:
                        fh.seek(128)
                        pre = fh.read(4)
                        has_preamble = (pre == b'DICM')
                except Exception:
                    has_preamble = False

                ds = None
                try:
                    if specific is not None:
                        try:
                            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True, specific_tags=specific)
                        except TypeError:
                            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                    else:
                        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                except Exception:
                    results.append(None)
                    continue

                img_type = getattr(ds, "ImageType", [])
                if not all([item in img_type for item in include_types]):
                    results.append(None)
                    continue
                if any([item in img_type for item in exclude_types]):
                    results.append(None)
                    continue

                rec = {
                    "path": fp,
                    "patient": getattr(ds, "PatientID", "UNKNOWN_MRN"),
                    "study_uid": str(getattr(ds, "StudyInstanceUID", None)),
                    "series_uid": str(getattr(ds, "SeriesInstanceUID", None)),
                    "series_desc": getattr(ds, "SeriesDescription", "UNKNOWN_SERIES"),
                    "series_number": int(getattr(ds, "SeriesNumber", 0)),
                }
                results.append(rec)
            except Exception:
                results.append(None)
    except Exception:
        # If import fails, return a list of None placeholders matching inputs
        results = [None] * len(filepaths)
    return results


def convert_defaultdict_to_dict(obj):
    if isinstance(obj, defaultdict):
        obj = {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        obj = {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj = [convert_defaultdict_to_dict(v) for v in obj]
    return obj


def reorder_hierarchy(hierarchy: dict, truncate_desc=5) -> dict:
    """Reorders a nested dictionary hierarchy[mrn][study_uid][desc]
    alphabetically by MRN, study_uid, and description (desc). The final
    keys (desc) are truncated by removing the first n characters.
    """
    studies_sorted_and_truncated = {}

    for mrn, study_dict in hierarchy.items():
        studies_sorted_and_truncated[mrn] = {}
        for study_uid, desc_dict in study_dict.items():
            sorted_items = sorted(desc_dict.items())
            truncated_desc_dict = {
                original_desc[truncate_desc:]: value
                for original_desc, value in sorted_items
            }
            studies_sorted_and_truncated[mrn][study_uid] = truncated_desc_dict

    studies_sorted_by_uid = {
        mrn: dict(sorted(study_dict.items()))
        for mrn, study_dict in studies_sorted_and_truncated.items()
    }

    fully_sorted_hierarchy = dict(sorted(studies_sorted_by_uid.items()))
    return fully_sorted_hierarchy


def build_hierarchy(dicom_records):
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    series_seen = set()

    for rec in dicom_records:
        if rec is None:
            continue
        series_uid = rec["series_uid"]
        if series_uid in series_seen:
            continue
        mrn = rec["patient"]
        study_uid = rec["study_uid"]
        desc = rec["series_desc"] or "UNKNOWN_SERIES"
        ser_num = rec["series_number"]
        series_seen.add(series_uid)
        desc = desc.strip() if isinstance(desc, str) else "UNKNOWN_SERIES"
        existing = hierarchy[mrn][study_uid]
        if desc in existing:
            suffix = 2
            new_desc = f"{desc} ({suffix})"
            while new_desc in existing:
                suffix += 1
                new_desc = f"{desc} ({suffix})"
            desc = new_desc
        desc = f"{ser_num:04d} {desc}"
        hierarchy[mrn][study_uid][desc].append(rec["path"])

    hierarchy = convert_defaultdict_to_dict(reorder_hierarchy(hierarchy, truncate_desc=5))
    return hierarchy

