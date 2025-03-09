"""
Contains the UI class that runs a Streamlit dashboard.
"""

import runpy
import sys
from importlib import resources

DASHBOARD_MODULE = "QuadratiK.ui"


class UI:
    """
    The UI class runs a Streamlit dashboard.
    Please see - https://quadratik.readthedocs.io/en/latest/user_guide/basic_usage.html#Initializing-the-Dashboard for more information.

    Examples
    ---------
        >>> from QuadratiK.ui import UI
        >>> UI().run()
    """

    def __init__(self) -> None:
        pass

    def run(self) -> None:
        """
        The function runs the Streamlit dashboard using runpy.
        """
        sys.argv = [
            "streamlit",
            "run",
            str(resources.files(DASHBOARD_MODULE).joinpath("_Introduction.py")),
            "--server.maxUploadSize",
            "20",
            "--theme.base",
            "light",
            "--theme.secondaryBackgroundColor",
            "#E5E4E2",
            "--theme.textColor",
            "#0e0e0e",
            "--browser.gatherUsageStats",
            "false",
        ]
        try:
            runpy.run_module("streamlit", run_name="__main__")
        except SystemExit as e:
            if e.code == 0:
                pass
            else:
                print(f"Dashboard exited with code {e.code}")
