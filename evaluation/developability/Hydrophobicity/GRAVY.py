"""Calculate CDR3 GRAVY for one sequence library."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import parse_args, run_sequence


def calculate_descriptors(record):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    return {"cdr3_gravy": ProteinAnalysis(record["generated_cdr3"]).gravy()}


def main():
    args = parse_args(__doc__)
    return run_sequence(args, "gravy", ["cdr3_gravy"], calculate_descriptors)


if __name__ == "__main__":
    raise SystemExit(main())
