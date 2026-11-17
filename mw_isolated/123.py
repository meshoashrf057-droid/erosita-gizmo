#!/usr/bin/env python3
"""
check_initial_mass_resolution.py

Report particle mass resolution for each PartType present in the initial
snapshot_000 of a GIZMO/Gadget HDF5 simulation.

For each particle type it prints:
  - particle count
  - mass in CODE UNITS
  - mass in Msun (from Header/UnitMass_in_g)

Fixed masses (typical for DM / stars) are flagged "const".
Gas masses are reported as median with [min, max].

Usage:
    python 123.py --snapshot-dir ./output
    python check_initial_mass_resolution.py --snapshot-dir ./output --sample 200000
"""

import argparse
import glob
import os
import sys

import numpy as np
import h5py

MSUN_G = 1.98892e33

# Best-effort labels; the true meaning depends on your compile-time config.
LABELS = {0: "gas", 1: "DM", 2: "disk", 3: "bulge", 4: "stars", 5: "BH"}


def find_snapshot(dirpath, num):
    pats = [f"snapshot_{num:03d}*.hdf5", f"snapshot_{num}*.hdf5"]
    files = []
    for p in pats:
        files.extend(glob.glob(os.path.join(dirpath, p)))
    files = sorted(set(files))
    if not files:
        sys.exit(f"No snapshot_{num:03d}*.hdf5 found in {dirpath!r}")
    return files


def read_masses(files, pt, sample=None):
    """Return 1-D array of masses (code units) across all files, or None."""
    chunks = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            path = f"PartType{pt}/Masses"
            if path not in f:
                continue
            d = f[path]
            n = len(d)
            if sample and n > sample:
                idx = np.sort(np.random.choice(n, sample, replace=False))
                chunks.append(d[idx])
            else:
                chunks.append(d[()])
    if not chunks:
        return None
    arr = np.concatenate(chunks)
    if sample and len(arr) > sample:
        idx = np.sort(np.random.choice(len(arr), sample, replace=False))
        arr = arr[idx]
    return arr


def main():
    ap = argparse.ArgumentParser(
        description="Report particle mass resolution in the initial snapshot_000.")
    ap.add_argument("--snapshot-dir", default="./output",
                    help="directory containing snapshot_000*.hdf5 (default: ./output)")
    ap.add_argument("--snapshot-number", type=int, default=0,
                    help="snapshot number to inspect (default: 0)")
    ap.add_argument("--sample", type=int, default=None,
                    help="randomly sample at most N particles per PartType "
                         "(counts still use Header/NumPart_Total)")
    args = ap.parse_args()

    files = find_snapshot(args.snapshot_dir, args.snapshot_number)

    # ---------------- Header ----------------
    with h5py.File(files[0], "r") as f:
        if "Header" not in f:
            sys.exit(f"No /Header in {files[0]}")
        H          = f["Header"].attrs
        npt        = H.get("NumPart_Total")
        mass_table = H.get("MassTable")
        unit_mass  = float(H.get("UnitMass_in_g", 1.0))
        hparam     = float(H.get("HubbleParam", 1.0))
        z          = float(H.get("Redshift", 0.0))
        boxsize    = H.get("BoxSize", None)
        parttypes  = sorted(
            int(k[8:]) for k in f.keys()
            if k.startswith("PartType") and k[8:].isdigit()
        )

    if not parttypes:
        sys.exit("No PartType* groups found in snapshot.")

    # Gadget/GIZMO convention: physical mass [g] = code_mass * UnitMass_in_g
    mass_to_msun = unit_mass / MSUN_G

    # ---------------- Header echo ----------------
    print(f"Snapshot {args.snapshot_number:03d}  ({len(files)} file(s))")
    for fn in files:
        print(f"  {fn}")
    print(f"Redshift z     = {z:.6f}")
    print(f"BoxSize (code) = {boxsize}")
    print(f"HubbleParam h  = {hparam:.6f}")
    print(f"UnitMass_in_g  = {unit_mass:.6e}")
    print(f"  => 1 code mass = {mass_to_msun:.6e} Msun")
    if hparam > 0 and abs(mass_to_msun / (1e10 / hparam) - 1.0) < 1e-3:
        print("  NOTE: this looks like the common 1e10 Msun/h unit;")
        print(f"        the Msun column below is in Msun/h -- divide by {hparam:.4f}")
        print("        for true Msun.")
    elif abs(mass_to_msun / 1e10 - 1.0) < 1e-3:
        print("  NOTE: this looks like the common 1e10 Msun unit (no h).")
    print()

    # ---------------- Per-type table ----------------
    header_line = (f"{'PartType':>8}  {'Label':<6}  {'Count':>12}  "
                   f"{'Mass [code]':<34}  {'Mass [Msun]':<34}")
    print(header_line)
    print("-" * 118)

    for pt in parttypes:
        label = LABELS.get(pt, "?")

        # particle count
        if npt is not None and pt < len(npt):
            n = int(npt[pt])
        else:
            n = 0

        arr = read_masses(files, pt, sample=args.sample)

        # Fall back to Header/MassTable if no per-particle masses stored
        if arr is None or len(arr) == 0:
            if (mass_table is not None and pt < len(mass_table)
                    and float(mass_table[pt]) > 0):
                m = float(mass_table[pt])
                med = lo = hi = m
                const = True
            else:
                print(f"{pt:>8}  {label:<6}  {n:>12d}  (no mass data)")
                continue
        else:
            med = float(np.median(arr))
            lo  = float(np.min(arr))
            hi  = float(np.max(arr))
            const = (hi - lo) <= 1e-9 * max(abs(med), 1e-30)

        if const:
            code_str = f"{med:.6e}  (const)"
            msun_str = f"{med * mass_to_msun:.6e}  (const)"
        else:
            code_str = f"med {med:.4e}  [{lo:.3e}, {hi:.3e}]"
            msun_str = (f"med {med * mass_to_msun:.4e}  "
                        f"[{lo * mass_to_msun:.3e}, {hi * mass_to_msun:.3e}]")

        print(f"{pt:>8}  {label:<6}  {n:>12d}  {code_str:<34}  {msun_str:<34}")

    print()
    print("Mass resolution = typical (median) particle mass of each type.")
    print("'const' means every particle of that type has the same mass --")
    print("for DM/stars that constant IS the mass resolution.")
    print("Gas masses vary; the median and [min, max] bracket the range.")


if __name__ == "__main__":
    main()