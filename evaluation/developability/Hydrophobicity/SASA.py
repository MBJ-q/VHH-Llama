"""Calculate total and hydrophobic SASA from one library of VHH structures."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_pyrosetta_options, parse_args, run_structure

TOTAL_SASA_METRIC = None
HYDROPHOBIC_SASA_METRIC = None
METRIC_FIELDS = [
    "n_residues", "total_sasa", "sasa_per_residue",
    "hydrophobic_sasa", "hydrophobic_sasa_fraction",
]


def initialize_worker(options):
    global TOTAL_SASA_METRIC, HYDROPHOBIC_SASA_METRIC
    import pyrosetta

    pyrosetta.init(options)
    metrics = pyrosetta.rosetta.core.simple_metrics.metrics
    TOTAL_SASA_METRIC = metrics.SasaMetric()
    TOTAL_SASA_METRIC.set_sasa_metric_mode("all_sasa")
    HYDROPHOBIC_SASA_METRIC = metrics.SasaMetric()
    HYDROPHOBIC_SASA_METRIC.set_sasa_metric_mode("hydrophobic_sasa")


def calculate_descriptors(pdb_path):
    import pyrosetta

    pose = pyrosetta.io.pose_from_pdbstring(pdb_path.read_text())
    n_residues = sum(pose.residue(index).is_protein() for index in range(1, pose.total_residue() + 1))
    total = float(TOTAL_SASA_METRIC.calculate(pose))
    hydrophobic = float(HYDROPHOBIC_SASA_METRIC.calculate(pose))
    if not n_residues or total <= 0:
        raise ValueError("Structure must contain protein residues and positive total SASA")
    return {
        "n_residues": n_residues, "total_sasa": total,
        "sasa_per_residue": total / n_residues, "hydrophobic_sasa": hydrophobic,
        "hydrophobic_sasa_fraction": hydrophobic / total,
    }


def main():
    args = parse_args(__doc__, structure=True, configure=add_pyrosetta_options)
    return run_structure(args, "sasa", METRIC_FIELDS, calculate_descriptors,
                         initialize_worker, (args.pyrosetta_options,))


if __name__ == "__main__":
    raise SystemExit(main())
