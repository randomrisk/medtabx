"""Skip tests whose sample data are not present.

The repository does not contain medical sample files. Tests that read them
run after ``python examples/data/download_samples.py``, or when a local
fixture folder exists at ``examples/data/ads_case``; otherwise they are
skipped.
"""

import inspect
from pathlib import Path

import pytest

EXAMPLE_DATA = Path(__file__).resolve().parents[1] / "examples" / "data"

# marker text in the test code -> file or folder that must exist
REQUIRED = {
    "ads_case": EXAMPLE_DATA / "ads_case",
    "diabetes_sample": EXAMPLE_DATA / "tabular" / "diabetes_sample.csv",
    "brain_mri_sample": EXAMPLE_DATA / "nifti",
    "chest_xray_sample": EXAMPLE_DATA / "dicom",
    "eeg_sample": EXAMPLE_DATA / "eeg",
    "chb01_01": EXAMPLE_DATA / "eeg",
    "ecg_sample": EXAMPLE_DATA / "ecg",
    "fmri_timeseries": EXAMPLE_DATA / "fmri",
    "vitals_timeseries": EXAMPLE_DATA / "vitals",
}


def _code_text(obj, module, seen):
    """Source of a function plus the values of module globals it uses."""
    if obj is None or id(obj) in seen:
        return ""
    seen.add(id(obj))
    try:
        text = inspect.getsource(obj)
    except (OSError, TypeError):
        return ""
    code = getattr(obj, "__code__", None)
    for name in getattr(code, "co_names", ()):
        value = getattr(module, name, None)
        if isinstance(value, (str, Path)):
            text += "\n" + str(value)
        elif inspect.isfunction(value) and value.__module__ == module.__name__:
            text += _code_text(value, module, seen)
    return text


def pytest_collection_modifyitems(config, items):
    for item in items:
        module = item.module
        seen = set()
        text = _code_text(getattr(item, "function", None), module, seen)
        defs = item.session._fixturemanager
        for name in item.fixturenames:
            for fixdef in defs.getfixturedefs(name, item) or ():
                text += _code_text(fixdef.func, module, seen)
        missing = [str(path.relative_to(EXAMPLE_DATA.parents[1]))
                   for marker, path in REQUIRED.items()
                   if marker in text and not path.exists()]
        if missing:
            item.add_marker(pytest.mark.skip(
                reason="sample data not present: " + ", ".join(sorted(set(missing)))))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Report a test as skipped when TabPFN weights need a license acceptance."""
    outcome = yield
    report = outcome.get_result()
    if (call.excinfo is not None
            and call.excinfo.typename == "TabPFNLicenseError"):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0,
                           "Skipped: TabPFN license not accepted on this machine")
