from __future__ import annotations

import unittest

from pymatgen.util.testing import PymatgenTest

from maml.data._url import FigshareSource, URLSource


class URLSourceTest(PymatgenTest):
    def test_get(self):
        source = URLSource()
        df = source.get("https://ndownloader.figshare.com/files/13007075")
        assert df.shape == (408, 29)


class FigshareSourceTest(PymatgenTest):
    def test_get(self):
        source = FigshareSource()
        df = source.get(12978425)
        assert df.shape == (1929, 81)


if __name__ == "__main__":
    unittest.main()
