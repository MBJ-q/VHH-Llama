"""Calculate CDR3 SAP from one library of IMGT-numbered VHH structures."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_pyrosetta_options, parse_args, run_structure

METRIC_FIELDS = ["cdr3_length", "sap_cdr3_score", "sap_cdr3_per_residue"]


def initialize_worker(options):
    import pyrosetta

    pyrosetta.init(options)


def calculate_descriptors(pdb_path):
    import pyrosetta

    pose = pyrosetta.io.pose_from_pdbstring(pdb_path.read_text())
    pdb_info = pose.pdb_info()
    indices = [
        index for index in range(1, pose.total_residue() + 1)
        if pose.residue(index).is_protein()
        and pdb_info.chain(index) == "H" and 105 <= pdb_info.number(index) <= 117
    ]
    if not indices:
        raise ValueError("No protein residues found in IMGT H105-H117")
    selectors = pyrosetta.rosetta.core.select.residue_selector
    cdr3_selector = selectors.ResidueIndexSelector()
    for index in indices:
        cdr3_selector.append_index(index)
    full_pose = selectors.TrueResidueSelector()
    # Retain the whole pose as the SAP contribution and SASA context (5 A default).
    metric = pyrosetta.rosetta.core.pack.guidance_scoreterms.sap.SapScoreMetric(
        cdr3_selector, full_pose, full_pose,
    )
    score = float(metric.calculate(pose))
    return {
        "cdr3_length": len(indices), "sap_cdr3_score": score,
        "sap_cdr3_per_residue": score / len(indices),
    }


def main():
    args = parse_args(__doc__, structure=True, configure=add_pyrosetta_options)
    return run_structure(args, "sap", METRIC_FIELDS, calculate_descriptors,
                         initialize_worker, (args.pyrosetta_options,))


if __name__ == "__main__":
    raise SystemExit(main())
