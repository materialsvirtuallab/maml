import unittest

from maml.utils._data_split import ShuffleSplitter


class TestShuffleSplit(unittest.TestCase):
    def test_split(self):
        ss = ShuffleSplitter()
        mat_ids = list(range(100))
        train, val, test = ss.split(mat_ids)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(test), 10)
        self.assertTrue(set(range(100)) == {*train, *val, *test})

        ss2 = ShuffleSplitter(ratios="90/10")
        train, test = ss2.split(mat_ids)
        self.assertEqual(len(train), 90)
        self.assertEqual(len(test), 10)

        ss2 = ShuffleSplitter(ratios="80:10:5:5", delim=":")
        splitted_ids = ss2.split(mat_ids)
        self.assertListEqual([len(i) for i in splitted_ids], [80, 10, 5, 5])


if __name__ == "__main__":
    unittest.main()
