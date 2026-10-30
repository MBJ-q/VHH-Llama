"""Calculate TNP PPC/PNC from one library of IMGT-numbered VHH structures."""
import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import parse_args, run_structure

CREATE_ANNOTATION = None
METRIC_FIELDS = ["ppc", "ppc_flag", "pnc", "pnc_flag"]


def initialize_worker():
    global CREATE_ANNOTATION
    from theraprofnano.Hydrophobicity_and_Charge_Profiler.Hydrophobicity_and_Charge_Assigner import CreateAnnotation

    if shutil.which("psa") is None:
        raise FileNotFoundError("TNP psa executable was not found in PATH")
    CREATE_ANNOTATION = CreateAnnotation


def is_hydrogen_atom(line):
    if line[:6].strip() not in {"ATOM", "HETATM"}:
        return False
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element == "H"
    return line[12:16].strip().upper().lstrip("123").startswith("H")


def assign_flag(value, amber, red):
    return "red" if value > red else "amber" if value >= amber else "green"


def calculate_descriptors(pdb_path):
    with tempfile.TemporaryDirectory(prefix="tnp_ppc_pnc_") as directory:
        clean_pdb = Path(directory) / "input.pdb"
        with pdb_path.open() as source, clean_pdb.open("w") as destination:
            for line in source:
                if not is_hydrogen_atom(line):
                    destination.write(line)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            annotation = CREATE_ANNOTATION(0, 7.4, str(clean_pdb), "IG", "imgt", verbose=False)
        ppc = float(annotation[0]["Patch_Pos_Charge_CDR"])
        pnc = float(annotation[0]["Patch_Neg_Charge_CDR"])
    return {
        "ppc": ppc, "ppc_flag": assign_flag(ppc, 0.39, 1.18),
        "pnc": pnc, "pnc_flag": assign_flag(pnc, 1.47, 1.88),
    }


def main():
    args = parse_args(__doc__, structure=True)
    return run_structure(args, "ppc_pnc", METRIC_FIELDS, calculate_descriptors, initialize_worker)


if __name__ == "__main__":
    raise SystemExit(main())
