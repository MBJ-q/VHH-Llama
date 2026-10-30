"""Shared single-library input, command-line arguments, and CSV output."""
from __future__ import annotations

import argparse
import csv
import math
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE_INPUT = PROJECT_ROOT / "VHH_libraries/VHH-Llama/generated_cdr3.csv"
DEFAULT_STRUCTURE_INPUT = PROJECT_ROOT / "VHH_libraries/VHH-Llama/predicted_structures"
DEFAULT_FRAMEWORKS = PROJECT_ROOT / "examples/therapeutic_vhh_scaffolds.csv"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
SEQUENCE_FIELDS = [
    "row_index", "sample_index", "sub_library", "generated_cdr3",
    "cdr3_length", "vhh_length",
]
STRUCTURE_FIELDS = ["row_index", "sample_index", "pdb_path"]
STATUS_FIELDS = ["status", "error_message"]
PDB_NAME_PATTERN = re.compile(r"row_(\d+)_sample_(\d+)\.pdb")


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args(description, structure=False, configure=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input", type=project_path,
        default=DEFAULT_STRUCTURE_INPUT if structure else DEFAULT_SEQUENCE_INPUT,
        help="One library: CSV file or structure directory. Relative paths use the repository root.",
    )
    parser.add_argument("--output", type=project_path, help="Output CSV; existing file is overwritten.")
    parser.add_argument("--limit", type=positive_int, help="Calculate only the first N records.")
    if structure:
        parser.add_argument("--workers", type=positive_int, default=1, help="CPU worker processes (default: 1).")
    else:
        parser.add_argument(
            "--frameworks", type=project_path,
            help="Framework CSV for inputs without L_seq/R_seq; published libraries use the example scaffolds.",
        )
    if configure:
        configure(parser)
    return parser.parse_args()


def output_path(args, metric, structure=False):
    library = args.input.name
    if structure and library == "predicted_structures":
        library = args.input.parent.name
    elif not structure:
        library = args.input.parent.name if args.input.name in (
            "generated_cdr3.csv", "structure_prediction_input.csv",
        ) else args.input.stem
    return args.output or PROJECT_ROOT / "outputs/developability" / library / f"{metric}_values.csv"


def clean_sequence(value):
    sequence = value.strip().upper()
    if not sequence or set(sequence) - AMINO_ACIDS:
        raise ValueError(f"Expected a non-empty sequence of canonical amino acids: {value!r}")
    return sequence


def iter_sequences(args):
    with args.input.open(newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"row_index", "sample_index", "generated_cdr3"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(f"Input CSV requires columns: {', '.join(sorted(required))}")
        inline_frameworks = {"L_seq", "R_seq"}.issubset(rows.fieldnames or [])
        frameworks = None
        if not inline_frameworks:
            frameworks_path = args.frameworks
            if frameworks_path is None:
                published_inputs = {
                    PROJECT_ROOT / "VHH_libraries" / library / filename
                    for library in ("IgLM", "VHH-Llama")
                    for filename in ("generated_cdr3.csv", "structure_prediction_input.csv")
                }
                if args.input.resolve() not in published_inputs:
                    raise ValueError("Input requires L_seq/R_seq columns or --frameworks.")
                frameworks_path = DEFAULT_FRAMEWORKS
            with frameworks_path.open(newline="") as frameworks_handle:
                frameworks = {
                    index: (row.get("name") or f"row_{index}",
                            clean_sequence(row["L_seq"]), clean_sequence(row["R_seq"]))
                    for index, row in enumerate(csv.DictReader(frameworks_handle))
                }
        for row in islice(rows, args.limit):
            row_index = int(row["row_index"])
            if inline_frameworks:
                name = row.get("name") or f"row_{row_index}"
                left, right = clean_sequence(row["L_seq"]), clean_sequence(row["R_seq"])
            else:
                name, left, right = frameworks[row_index]
            cdr3 = clean_sequence(row["generated_cdr3"])
            vhh = left + cdr3 + right
            yield {
                "row_index": row_index, "sample_index": int(row["sample_index"]),
                "sub_library": name, "generated_cdr3": cdr3,
                "cdr3_length": len(cdr3), "vhh_length": len(vhh), "vhh": vhh,
            }


def discover_structures(root, limit=None):
    if not root.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {root}; supply --input.")
    items = []
    for path in root.glob("row_*/*.pdb"):
        match = PDB_NAME_PATTERN.fullmatch(path.name)
        if match:
            items.append({
                "row_index": int(match[1]), "sample_index": int(match[2]),
                "pdb_path": str(path.resolve()),
            })
    items.sort(key=lambda item: (item["row_index"], item["sample_index"]))
    if not items:
        raise ValueError(f"No row_*/row_<index>_sample_<index>.pdb files found in {root}")
    return items[:limit] if limit else items


def calculate_record(item, calculator, structure=False):
    result = {key: value for key, value in item.items() if key != "vhh"}
    try:
        values = calculator(Path(item["pdb_path"]) if structure else item)
        if any(isinstance(value, float) and not math.isfinite(value) for value in values.values()):
            raise ValueError("Calculation returned a non-finite value")
        result.update(values, status="success", error_message="")
    except Exception as error:
        result.update(status="failed", error_message=f"{type(error).__name__}: {error}")
    return result


def write_results(path, fields, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = failed = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
            count += 1
            failed += result["status"] == "failed"
    print(f"Wrote {count} records ({failed} failed) to {path}")
    return 1 if failed else 0


def run_sequence(args, metric, fields, calculator):
    results = (calculate_record(item, calculator) for item in iter_sequences(args))
    return write_results(output_path(args, metric), SEQUENCE_FIELDS + fields + STATUS_FIELDS, results)


def add_pyrosetta_options(parser):
    parser.add_argument("--pyrosetta-options", default="-mute all -ignore_unrecognized_res true")


def run_structure(args, metric, fields, calculator, initializer, initargs=()):
    items = discover_structures(args.input, args.limit)
    calculate = partial(calculate_record, calculator=calculator, structure=True)
    path = output_path(args, metric, structure=True)
    columns = STRUCTURE_FIELDS + fields + STATUS_FIELDS
    if args.workers == 1:
        initializer(*initargs)
        return write_results(path, columns, map(calculate, items))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initializer, initargs=initargs) as pool:
        return write_results(path, columns, pool.map(calculate, items, chunksize=1))
