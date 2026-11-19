"""
Compare the SN parameters of two trainings, matching SNe by name.

Usage: python -m saltshaker.validation.comparesnparams new/salt3train_snparams.txt old/salt3train_snparams.txt

Exits with status 1 when the files have different SNe or any parameter differs beyond
the tolerances.
"""

import argparse
import sys

import numpy as np
import pandas as pd

# Absolute tolerances; each parameter also has the relative tolerance
ATOL = {"x0": 0.0, "x1": 0.01, "c": 0.001, "t0": 0.01}
RTOL = 1e-3


def readsnparams(filename):
    """SN parameters from a salt3train_snparams.txt file, indexed by SN name."""
    with open(filename) as fin:
        columns = fin.readline().lstrip("#").split()
    table = pd.read_csv(filename, sep=r"\s+", comment="#", names=columns, dtype={"SN": str})
    # Reference outputs can repeat each row once per SN list
    table = table.drop_duplicates()
    conflicting = table["SN"][table["SN"].duplicated()].unique()
    if conflicting.size:
        raise ValueError(f"{filename} has conflicting rows for {', '.join(conflicting[:10])}")
    return table.set_index("SN")


def comparesnparams(new, old, atol=ATOL, rtol=RTOL):
    """
    Differences between two tables of SN parameters.

    Returns
    -------
    missing : list of str
        Messages for SNe in only one of the tables.
    differing : dict
        For each parameter in atol, the SNe beyond tolerance as (SN, new, old) tuples,
        largest relative difference first.
    """
    missing = [f"{len(names)} SNe only in the {which} training, e.g. {', '.join(names[:5])}"
               for names, which in [(new.index.difference(old.index), "new"),
                                    (old.index.difference(new.index), "old")] if len(names)]
    common = new.index.intersection(old.index)
    differing = {}
    for column, columnatol in atol.items():
        a, b = new.loc[common, column].to_numpy(float), old.loc[common, column].to_numpy(float)
        bad = ~np.isclose(a, b, rtol=rtol, atol=columnatol)
        relative = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)
        order = np.argsort(-relative)
        differing[column] = [(common[i], a[i], b[i]) for i in order if bad[i]]
    return missing, differing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("new", help="salt3train_snparams.txt of the training being checked")
    parser.add_argument("old", help="salt3train_snparams.txt of the reference training")
    args = parser.parse_args(argv)

    new, old = readsnparams(args.new), readsnparams(args.old)
    missing, differing = comparesnparams(new, old)
    print(f"{len(new)} SNe in {args.new}, {len(old)} in {args.old}")
    for message in missing:
        print(message)
    for column, rows in differing.items():
        print(f"{column}: {len(rows)} SNe beyond tolerance (atol {ATOL[column]}, rtol {RTOL})")
        for sn, a, b in rows[:5]:
            print(f"    {sn}: {a:.6g} vs {b:.6g}")
    return 1 if missing or any(differing.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
