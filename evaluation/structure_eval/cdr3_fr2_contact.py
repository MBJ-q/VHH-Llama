"""Calculate CDR3--FR2 soft residue contacts for one IMGT-numbered VHH library.

Use chain H, FR2 positions 39--55 and CDR3 positions 105--117, including
insertion codes. Exclude the first and last two observed CDR3 residues.
For each remaining residue, find its closest heavy-atom distance d to FR2:
    contact_nres = sum(1 / (1 + exp((d - 4.5) / 0.4)))
Distances are in angstroms. Outputs contain calculated values only.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import islice
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "developability"))
from _common import PROJECT_ROOT, discover_structures, parse_args, project_path

CHAIN_ID = "H"
IMGT_FR2_START = 39
IMGT_FR2_END = 55
IMGT_CDR3_START = 105
IMGT_CDR3_END = 117
CDR3_STEM_SIZE = 2
CONTACT_MIDPOINT = 4.5
CONTACT_WIDTH = 0.4
OUTPUT_FIELDS = (
    "model", "library", "row_index", "sample_index", "generated_cdr3",
    "source_cdr3_length", "structure_cdr3_length", "nonstem_cdr3_length",
    "contact_nres", "status", "error", "structure_file",
)


def _element_from_pdb_line(line: str, atom_name: str) -> str:
    element = line[76:78].strip().upper()
    if element:
        return element
    letters = "".join(character for character in atom_name if character.isalpha())
    return letters[:1].upper()


def read_imgt_regions(pdb_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read FR2 and CDR3 heavy atoms from the first model of chain H."""
    fr2: dict[tuple[int, str], list[np.ndarray]] = {}
    cdr3: dict[tuple[int, str], list[np.ndarray]] = {}
    in_first_model = True
    model_seen = False

    with pdb_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = line[0:6].strip()
            if record == "MODEL":
                if model_seen:
                    in_first_model = False
                else:
                    model_seen = True
                continue
            if record == "ENDMDL" and model_seen:
                break
            if not in_first_model or record != "ATOM":
                continue
            if line[21].strip() != CHAIN_ID:
                continue
            if line[16].strip() not in {"", "A"}:
                continue

            try:
                residue_number = int(line[22:26])
                coordinate = np.asarray(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                    dtype=float,
                )
            except ValueError:
                continue

            atom_name = line[12:16].strip()
            element = _element_from_pdb_line(line, atom_name)
            if element in {"H", "D"}:
                continue

            residue_id = (residue_number, line[26].strip())
            if IMGT_FR2_START <= residue_number <= IMGT_FR2_END:
                fr2.setdefault(residue_id, []).append(coordinate)
            elif IMGT_CDR3_START <= residue_number <= IMGT_CDR3_END:
                cdr3.setdefault(residue_id, []).append(coordinate)

    if not fr2:
        raise ValueError("no IMGT FR2 residues 39-55 found")
    if not cdr3:
        raise ValueError("no IMGT CDR3 residues 105-117 found")
    return (
        [np.asarray(atoms, dtype=float) for atoms in fr2.values()],
        [np.asarray(atoms, dtype=float) for atoms in cdr3.values()],
    )


def calculate_contact_nres(
    fr2_residues: list[np.ndarray], cdr3_residues: list[np.ndarray],
) -> tuple[int, float]:
    cdr3_nonstem = cdr3_residues[CDR3_STEM_SIZE:-CDR3_STEM_SIZE]
    fr2_atoms = np.concatenate(fr2_residues, axis=0)
    contact_nres = 0.0
    for residue_atoms in cdr3_nonstem:
        differences = residue_atoms[:, None, :] - fr2_atoms[None, :, :]
        minimum_distance = math.sqrt(
            float(np.min(np.einsum("ijk,ijk->ij", differences, differences)))
        )
        contact_nres += 1.0 / (
            1.0 + math.exp((minimum_distance - CONTACT_MIDPOINT) / CONTACT_WIDTH)
        )
    return len(cdr3_nonstem), contact_nres


def library_name(root: Path) -> str:
    return root.parent.name if root.name == "predicted_structures" else root.name


def load_tasks(root: Path, metadata: Path | None, limit: int | None) -> list[dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {root}")
    if metadata is None:
        published_input = root.parent / "structure_prediction_input.csv"
        if published_input.is_file():
            metadata = published_input

    if metadata is None:
        records = discover_structures(root, limit)
        return [
            {
                "model": library_name(root), "library": f"row_{item['row_index']}",
                "row_index": item["row_index"], "sample_index": item["sample_index"],
                "generated_cdr3": "", "source_cdr3_length": "",
                "structure_file": item["pdb_path"],
            }
            for item in records
        ]

    tasks = []
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        records = csv.DictReader(handle)
        if not {"row_index", "sample_index"}.issubset(records.fieldnames or []):
            raise ValueError("Metadata CSV requires row_index and sample_index")
        for record in islice(records, limit):
            row_index, sample_index = int(record["row_index"]), int(record["sample_index"])
            cdr3 = (record.get("generated_cdr3") or "").strip()
            tasks.append({
                "model": library_name(root), "library": f"row_{row_index}",
                "row_index": row_index, "sample_index": sample_index,
                "generated_cdr3": cdr3,
                "source_cdr3_length": record.get("length") or (len(cdr3) if cdr3 else ""),
                "structure_file": str((root / f"row_{row_index}" /
                                       f"row_{row_index}_sample_{sample_index}.pdb").resolve()),
            })
    return tasks


def calculate_task(task: dict) -> dict:
    result = {field: "" for field in OUTPUT_FIELDS}
    result.update(task)
    pdb_path = Path(task["structure_file"])
    if not pdb_path.is_file():
        result.update(status="missing_structure", error="structure file not found")
        return result
    try:
        fr2, cdr3 = read_imgt_regions(pdb_path)
        nonstem_length, contact_nres = calculate_contact_nres(fr2, cdr3)
        result.update(
            structure_cdr3_length=str(len(cdr3)), nonstem_cdr3_length=str(nonstem_length),
            contact_nres=f"{contact_nres:.8f}", status="ok", error="",
        )
    except Exception as error:
        result.update(status="structure_error", error=str(error))
    return result


def configure(parser):
    parser.add_argument(
        "--metadata", type=project_path,
        help="Optional input CSV; defaults to structure_prediction_input.csv next to the published structure directory.",
    )


def calculate_results(tasks, workers):
    if workers == 1:
        yield from map(calculate_task, tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(calculate_task, tasks, chunksize=1)


def main() -> int:
    args = parse_args(__doc__, structure=True, configure=configure)
    tasks = load_tasks(args.input, args.metadata, args.limit)
    output = args.output or (
        PROJECT_ROOT / "outputs/structure_eval" / library_name(args.input) /
        "cdr3_fr2_contact_values.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    count = failed = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for result in calculate_results(tasks, args.workers):
            writer.writerow(result)
            count += 1
            failed += result["status"] != "ok"
    print(f"Wrote {count} records ({failed} without a score) to {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
