from pymatgen.util.testing import PymatgenTest

from maml.data.url import URLSource, FigshareSource


class URLSourceTest(PymatgenTest):

    def test_get(self):
        source = URLSource()
        df = source.get("https://ndownloader.figshare.com/files/13007075")
        assert df.shape == (408, 29)


class FigshareSourceTest(PymatgenTest):

    def test_get(self):
        source = FigshareSource()
        df = source.get(12978425)
        print(df)
        assert df.shape == (1929, 81)
