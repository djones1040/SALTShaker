import os
import tempfile
import unittest

from saltshaker.validation.comparesnparams import comparesnparams, main, readsnparams

HEADER = "# SN x0 x1 c t0 SIM_x0 SIM_x1 SIM_c SIM_t0 SALT2_x0 SALT2_x1 SALT2_c SALT2_t0\n"
ROWS = {
    "2006D": "4.17e-02 -1.4409 -0.0190 53755.8828",
    "06D2bk": "7.98e-06 2.1728 0.1142 54012.0000",
    "1346966": "3.74e-05 0.4283 -0.0950 57407.5312",
}
SIM = " -99 -99 -99 -99 -99 -99 -99 -99"


class test_comparesnparams(unittest.TestCase):

    def write(self, rows, name):
        filename = os.path.join(self.tmpdir.name, name)
        with open(filename, "w") as fout:
            fout.write(HEADER + "".join(f"{sn} {values}{SIM}\n" for sn, values in rows))
        return filename

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.reference = self.write(list(ROWS.items()), "old.txt")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_same_parameters_pass(self):
        """Repeated and reordered rows with the same parameters compare equal."""
        rows = list(reversed(ROWS.items())) * 3
        self.assertEqual(main([self.write(rows, "new.txt"), self.reference]), 0)

    def test_parameter_change_fails(self):
        """A change in x1 beyond its tolerance is reported for that SN only."""
        rows = dict(ROWS, **{"06D2bk": "7.98e-06 2.2228 0.1142 54012.0000"})
        new = self.write(rows.items(), "new.txt")
        missing, differing = comparesnparams(readsnparams(new), readsnparams(self.reference))
        self.assertEqual(missing, [])
        self.assertEqual([sn for sn, _, _ in differing["x1"]], ["06D2bk"])
        self.assertEqual(differing["x0"] + differing["c"] + differing["t0"], [])
        self.assertEqual(main([new, self.reference]), 1)

    def test_missing_sn_fails(self):
        """SNe in only one training are reported."""
        new = self.write(list(ROWS.items())[1:], "new.txt")
        missing, _ = comparesnparams(readsnparams(new), readsnparams(self.reference))
        self.assertEqual(len(missing), 1)
        self.assertIn("2006D", missing[0])
        self.assertEqual(main([new, self.reference]), 1)

    def test_conflicting_rows_raise(self):
        """Two different rows for one SN are an error."""
        rows = list(ROWS.items()) + [("2006D", "4.17e-02 -1.0000 -0.0190 53755.8828")]
        with self.assertRaisesRegex(ValueError, "2006D"):
            readsnparams(self.write(rows, "new.txt"))


if __name__ == "__main__":
    unittest.main()
