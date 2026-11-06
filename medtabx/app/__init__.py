"""Browser interface over the typed contract.

    python -m medtabx.app

See :mod:`medtabx.app.ui` for the flow and :mod:`medtabx.app.scan` for the
folder scanner that proposes a schema.
"""

from medtabx.app.scan import ScanResult, scan_study

__all__ = ["ScanResult", "scan_study"]
