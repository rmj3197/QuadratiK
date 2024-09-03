"""
Contains the UI class is a user interface class that runs a Streamlit
dashboard using asyncio.
"""

import asyncio
import sys
from importlib import resources

import nest_asyncio
from streamlit.web import cli as stcli

nest_asyncio.apply()


DASHBOARD_MODULE = "QuadratiK.ui"


class UI:
    """
    The UI class is a user interface class that runs a Streamlit dashboard using asyncio.

    Examples
    ---------
        >>> from QuadratiK.ui import UI
        >>> UI().run()
    """

    def __init__(self) -> None:
        pass

    async def main(self) -> None:
        """
        The `main` function runs a Streamlit dashboard by executing a command-line command.
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
        sys.exit(stcli.main())

    def run(self) -> None:
        """
        The function runs the main function asynchronously using the asyncio library in Python.
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.main())
