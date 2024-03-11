"""
Checks if QuadratiK can be imported correctly along with submodules
"""

import unittest
import importlib
import QuadratiK


class TestQuadratiKModule(unittest.TestCase):

    def test_version(self):
        from QuadratiK import __version__

        self.assertTrue(isinstance(__version__, str))

    def test_dir(self):
        import QuadratiK

        self.assertIsNotNone(dir(QuadratiK))

    def test_submodules_exist(self):
        from QuadratiK import submodules

        for submodule in submodules:
            self.assertIsNotNone(getattr(QuadratiK, submodule, None))

    def test_invalid_attribute(self):
        with self.assertRaises(AttributeError):
            import QuadratiK

            QuadratiK.__some__
