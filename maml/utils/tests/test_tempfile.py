from glob import glob
import os
import unittest

from maml.utils._tempfile import MultiScratchDir
from monty.tempfile import ScratchDir


class TestMultiScratchDir(unittest.TestCase):
    def test_multi(self):
        with ScratchDir("."):
            with MultiScratchDir(".", n_dirs=4, copy_to_current_on_exit=True) as dirs:
                self.assertTrue(len(dirs) == 4)
                for d in dirs:
                    os.chdir(d)
                    os.system("touch tempfile")
                    os.chdir("..")
            tempfiles = set(glob("tempfile*"))
            self.assertEqual(set(["tempfile_%d" % i for i in range(4)]), tempfiles)


if __name__ == "__main__":
    unittest.main()
