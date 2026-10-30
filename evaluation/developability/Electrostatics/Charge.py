"""Calculate full-VHH net charge at pH 7.4 for one sequence library."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import parse_args, run_sequence


def calculate_descriptors(record):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    return {"charge_pH7_4": ProteinAnalysis(record["vhh"]).charge_at_pH(7.4)}


def main():
    args = parse_args(__doc__)
    return run_sequence(args, "charge", ["charge_pH7_4"], calculate_descriptors)


if __name__ == "__main__":
    raise SystemExit(main())
