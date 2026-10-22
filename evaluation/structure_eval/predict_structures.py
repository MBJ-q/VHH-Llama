from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSVS = (
    PROJECT_ROOT / "libraries" / "IgLM" / "structure_prediction_input.csv",
    PROJECT_ROOT / "libraries" / "VHH-Llama" / "structure_prediction_input.csv",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predicted_structures"
DEFAULT_GPUS = "0,1,2,3"
DEFAULT_WORKERS_PER_GPU = "5"
REQUIRED_COLUMNS = {"row_index", "sample_index", "generated_cdr3"}

# Frameworks in examples/therapeutic_vhh_scaffolds.csv, in CSV row order.
FRAMEWORKS = {
    0: (
        "EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYC",
        "WGQGTQVTVSS",
    ),
    1: (
        "EVQLVESGGGLVQPGGSLRLSCAASGRTYDAMGWLRQAPGKEREFVAAISGSGDDTYYADSVKGRFTISRDNSKNTLYLQMNSLRPEDTAVYYC",
        "WGQGTLVTVSS",
    ),
    2: (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYC",
        "RGQGTLVTVSS",
    ),
}


@dataclass(frozen=True)
class PredictionTask:
    row_index: int
    sample_index: int
    sequence: str

    @property
    def filename(self) -> str:
        return f"row_{self.row_index}_sample_{self.sample_index}.pdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict nanobody structures with NanoBodyBuilder2. By default, "
            "process both libraries/IgLM/structure_prediction_input.csv and "
            "libraries/VHH-Llama/structure_prediction_input.csv."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="*",
        default=DEFAULT_INPUT_CSVS,
        help=(
            "Optional input CSV paths containing row_index, sample_index, and "
            "generated_cdr3. Relative paths are resolved from the repository "
            "root. Each CSV's parent directory names its output subdirectory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output root (default: outputs/predicted_structures). PDBs are "
            "saved under <output-dir>/<library>/row_<n>/. Relative paths "
            "are resolved from the repository root."
        ),
    )
    parser.add_argument(
        "--gpus",
        default=DEFAULT_GPUS,
        help="Comma-separated physical GPU indices (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        default=DEFAULT_WORKERS_PER_GPU,
        help=(
            "One worker count for all GPUs, or comma-separated counts in "
            "--gpus order (default: 5)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Predict again even when the final PDB already exists",
    )
    parser.add_argument(
        "--row-indices",
        type=int,
        nargs="+",
        default=None,
        help="Only process these row_index values (default: all rows)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry samples listed in an existing row_<n>_failures.json file",
    )
    return parser.parse_args()


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--gpus must look like 0, 0,1, or 0,1,2,3") from error

    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")
    if any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU indices must be non-negative")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU indices must not be repeated")
    return gpus


def parse_workers_per_gpu(
    value: str,
    gpus: Sequence[int],
) -> dict[int, int]:
    try:
        worker_counts = [
            int(item.strip()) for item in value.split(",") if item.strip()
        ]
    except ValueError as error:
        raise ValueError(
            "--workers-per-gpu must be an integer or a comma-separated "
            "integer list"
        ) from error

    if len(worker_counts) == 1:
        worker_counts *= len(gpus)
    elif len(worker_counts) != len(gpus):
        raise ValueError(
            "--workers-per-gpu must provide one value or one value per GPU "
            "in --gpus order"
        )
    if any(worker_count <= 0 for worker_count in worker_counts):
        raise ValueError("--workers-per-gpu values must be greater than 0")
    return dict(zip(gpus, worker_counts))


def load_tasks(input_csv: Path) -> dict[int, list[PredictionTask]]:
    tasks_by_row = {row_index: [] for row_index in FRAMEWORKS}
    seen_filenames: set[str] = set()

    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(missing)}"
            )

        for csv_line, row in enumerate(reader, start=2):
            try:
                row_index = int(row["row_index"])
                sample_index = int(row["sample_index"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid row_index or sample_index at CSV line {csv_line}"
                ) from error

            if row_index not in FRAMEWORKS:
                raise ValueError(
                    f"Unsupported row_index {row_index} at CSV line {csv_line}; "
                    f"expected one of {sorted(FRAMEWORKS)}"
                )

            generated_cdr3 = (row["generated_cdr3"] or "").strip().upper()
            if not generated_cdr3:
                raise ValueError(f"Empty generated_cdr3 at CSV line {csv_line}")

            left_framework, right_framework = FRAMEWORKS[row_index]
            task = PredictionTask(
                row_index=row_index,
                sample_index=sample_index,
                sequence=left_framework + generated_cdr3 + right_framework,
            )
            if task.filename in seen_filenames:
                raise ValueError(f"Duplicate output filename: {task.filename}")
            seen_filenames.add(task.filename)
            tasks_by_row[row_index].append(task)

    for row_index, tasks in tasks_by_row.items():
        if not tasks:
            raise ValueError(f"Input CSV contains no sequences for row_{row_index}")
    return tasks_by_row


def failure_manifest_path(output_dir: Path, row_index: int) -> Path:
    return output_dir / f"row_{row_index}_failures.json"


def load_failure_records(output_dir: Path, row_index: int) -> dict[str, dict]:
    manifest_path = failure_manifest_path(output_dir, row_index)
    if not manifest_path.is_file():
        return {}

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        item["filename"]: item
        for item in manifest.get("failures", [])
        if "filename" in item
    }


def save_failure_records(
    output_dir: Path,
    row_index: int,
    records: list[dict],
) -> None:
    manifest_path = failure_manifest_path(output_dir, row_index)
    if not records:
        manifest_path.unlink(missing_ok=True)
        return

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"row_index": row_index, "failures": records},
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Saved failure manifest: {manifest_path}")


def prepare_library_directory(
    output_dir: Path,
    row_index: int,
    tasks: Sequence[PredictionTask],
    overwrite: bool,
    known_failures: set[str],
    retry_failed: bool,
) -> tuple[Path, list[PredictionTask], bool]:
    final_dir = output_dir / f"row_{row_index}"
    work_dir = output_dir / f".row_{row_index}_in_progress"
    expected_filenames = {task.filename for task in tasks}

    if final_dir.exists() and not final_dir.is_dir():
        raise ValueError(f"Library output path is not a directory: {final_dir}")
    if work_dir.exists() and not work_dir.is_dir():
        raise ValueError(f"Library work path is not a directory: {work_dir}")

    final_is_complete = final_dir.is_dir() and all(
        (final_dir / filename).is_file() or filename in known_failures
        for filename in expected_filenames
    )
    if final_is_complete and not overwrite and not retry_failed:
        return work_dir, [], True

    if final_dir.is_dir():
        if not work_dir.exists():
            final_dir.rename(work_dir)
        else:
            for path in final_dir.iterdir():
                os.replace(path, work_dir / path.name)
            final_dir.rmdir()
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        pending = list(tasks)
    else:
        pending = [task for task in tasks if not (work_dir / task.filename).is_file()]
    if not retry_failed:
        pending = [task for task in pending if task.filename not in known_failures]
    return work_dir, pending, False


def publish_library(
    output_dir: Path,
    row_index: int,
    work_dir: Path,
    tasks: Sequence[PredictionTask],
    failed_filenames: set[str],
) -> None:
    missing = [
        task.filename
        for task in tasks
        if not (work_dir / task.filename).is_file()
        and task.filename not in failed_filenames
    ]
    if missing:
        raise RuntimeError(
            f"row_{row_index} is incomplete; {len(missing)} expected PDB(s) are missing"
        )

    for partial_path in work_dir.glob(".*.partial"):
        partial_path.unlink()

    final_dir = output_dir / f"row_{row_index}"
    if final_dir.exists():
        raise FileExistsError(f"Cannot publish over existing directory: {final_dir}")
    work_dir.rename(final_dir)


def split_for_gpus(
    tasks: Sequence[PredictionTask],
    gpus: Sequence[int],
) -> dict[int, list[PredictionTask]]:
    return {gpu: list(tasks[position :: len(gpus)]) for position, gpu in enumerate(gpus)}


def predict_worker(
    gpu: int,
    worker_index: int,
    tasks: Sequence[PredictionTask],
    work_dir: Path,
    result_queue: mp.Queue,
) -> None:
    # CUDA must be restricted before importing torch/ImmuneBuilder in a spawned child.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    try:
        import torch
        from ImmuneBuilder import NanoBodyBuilder2

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA GPU {gpu} is not available")
        predictor = NanoBodyBuilder2()
    except Exception as error:
        result_queue.put(
            (
                "worker_init_error",
                len(tasks),
                gpu,
                worker_index,
                f"{type(error).__name__}: {error}",
            )
        )
        return

    for task in tasks:
        output_path = work_dir / task.filename
        partial_path = work_dir / f".{task.filename}.partial"

        try:
            prediction = predictor.predict({"H": task.sequence})
            prediction.save(str(partial_path))
            os.replace(partial_path, output_path)
            result_queue.put(("success", 1, gpu, worker_index, task.filename))
        except Exception as error:
            if partial_path.exists():
                partial_path.unlink()
            result_queue.put(
                (
                    "prediction_error",
                    1,
                    gpu,
                    worker_index,
                    f"{task.filename}: {type(error).__name__}: {error}",
                )
            )


def start_workers(
    gpu_tasks: dict[int, list[PredictionTask]],
    workers_by_gpu: dict[int, int],
    work_dir: Path,
    result_queue: mp.Queue,
    context: mp.context.BaseContext,
) -> list[mp.Process]:
    processes: list[mp.Process] = []

    for gpu, tasks in gpu_tasks.items():
        worker_count = workers_by_gpu[gpu]
        for worker_index in range(worker_count):
            worker_tasks = tasks[worker_index::worker_count]
            process = context.Process(
                target=predict_worker,
                args=(
                    gpu,
                    worker_index,
                    worker_tasks,
                    work_dir,
                    result_queue,
                ),
                name=f"gpu-{gpu}-worker-{worker_index}",
            )
            process.start()
            processes.append(process)

    return processes


def wait_for_predictions(
    processes: Sequence[mp.Process],
    result_queue: mp.Queue,
    total: int,
    row_index: int,
    tqdm,
) -> tuple[int, int, list[str], dict[str, str]]:
    completed = 0
    succeeded = 0
    failed = 0
    error_examples: list[str] = []
    failure_details: dict[str, str] = {}

    with tqdm(total=total, desc=f"Predicting row_{row_index}", unit="PDB") as progress:
        while completed < total:
            try:
                kind, count, gpu, worker_index, detail = result_queue.get(timeout=1)
            except queue.Empty:
                if any(process.is_alive() for process in processes):
                    continue
                missing = total - completed
                failed += missing
                completed += missing
                error_examples.append(
                    f"{missing} task(s) produced no result because workers exited"
                )
                progress.update(missing)
                break

            completed += count
            progress.update(count)

            if kind == "success":
                succeeded += count
            else:
                failed += count
                filename, separator, _ = detail.partition(":")
                if separator and filename.endswith(".pdb"):
                    failure_details[filename] = detail
                if len(error_examples) < 20:
                    error_examples.append(
                        f"GPU {gpu}, worker {worker_index}: {detail}"
                    )

    for process in processes:
        process.join()
    return succeeded, failed, error_examples, failure_details


def collect_failure_records(
    tasks: Sequence[PredictionTask],
    work_dir: Path,
    known_records: dict[str, dict],
    failure_details: dict[str, str],
) -> list[dict]:
    records = []
    for task in tasks:
        if (work_dir / task.filename).is_file():
            continue
        record = dict(known_records.get(task.filename, {}))
        record.update(
            {
                "filename": task.filename,
                "sample_index": task.sample_index,
                "error": failure_details.get(
                    task.filename,
                    record.get("error", "Prediction did not produce a PDB file"),
                ),
            }
        )
        records.append(record)
    return records


def predict_library(
    args: argparse.Namespace,
    input_csv: Path,
    output_dir: Path,
    gpus: Sequence[int],
    workers_by_gpu: dict[int, int],
    requested_rows: Sequence[int],
    tqdm,
) -> tuple[int, int, int]:
    tasks_by_row = load_tasks(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input CSV: {input_csv}")
    print(f"Output directory: {output_dir}")
    print(f"GPUs and workers: {workers_by_gpu}")
    context = mp.get_context("spawn")
    total_succeeded = 0
    total_skipped = 0

    total_failed = 0
    for row_index in requested_rows:
        library_tasks = tasks_by_row[row_index]
        known_failure_records = load_failure_records(output_dir, row_index)
        known_failures = (
            set() if args.retry_failed else set(known_failure_records)
        )
        work_dir, pending_tasks, already_complete = prepare_library_directory(
            output_dir,
            row_index,
            library_tasks,
            args.overwrite,
            known_failures,
            args.retry_failed,
        )

        if already_complete:
            total_skipped += len(library_tasks)
            print(f"row_{row_index}: complete library already exists; skipped")
            continue

        resumed = len(library_tasks) - len(pending_tasks)
        total_skipped += resumed
        print(
            f"row_{row_index}: total={len(library_tasks)}, "
            f"pending={len(pending_tasks)}, resumed={resumed}"
        )

        if pending_tasks:
            gpu_tasks = split_for_gpus(pending_tasks, gpus)
            for gpu in gpus:
                print(
                    f"row_{row_index}, GPU {gpu}: {len(gpu_tasks[gpu])} "
                    f"prediction(s), {workers_by_gpu[gpu]} worker(s)"
                )

            result_queue = context.Queue(
                maxsize=max(100, sum(workers_by_gpu.values()) * 4)
            )
            processes = start_workers(
                gpu_tasks,
                workers_by_gpu,
                work_dir,
                result_queue,
                context,
            )

            try:
                succeeded, failed, error_examples, failure_details = wait_for_predictions(
                    processes,
                    result_queue,
                    len(pending_tasks),
                    row_index,
                    tqdm,
                )
            except KeyboardInterrupt:
                for process in processes:
                    process.terminate()
                for process in processes:
                    process.join()
                raise

            total_succeeded += succeeded
            total_failed += failed
            if error_examples:
                print(f"row_{row_index} failure examples:", file=sys.stderr)
                for message in error_examples:
                    print(f"  - {message}", file=sys.stderr)
        else:
            failure_details = {}

        failure_records = collect_failure_records(
            library_tasks,
            work_dir,
            known_failure_records,
            failure_details,
        )
        save_failure_records(output_dir, row_index, failure_records)
        publish_library(
            output_dir,
            row_index,
            work_dir,
            library_tasks,
            {record["filename"] for record in failure_records},
        )
        print(
            f"row_{row_index}: library published; "
            f"PDBs={len(library_tasks) - len(failure_records)}, "
            f"failures={len(failure_records)}"
        )

    print(
        f"Completed: {total_succeeded}; skipped/resumed: {total_skipped}; "
        f"failed: {total_failed}"
    )
    return total_succeeded, total_skipped, total_failed


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def main() -> int:
    args = parse_args()

    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError(
            "tqdm is required. Install it in the NBB2 environment before running."
        ) from error

    input_csvs = [resolve_path(path) for path in args.input_csv]
    library_names = [path.parent.name for path in input_csvs]
    if len(set(library_names)) != len(library_names):
        raise ValueError(
            "Input CSVs must have distinct parent directory names to avoid "
            "overwriting library outputs"
        )
    for input_csv in input_csvs:
        if not input_csv.is_file():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    gpus = parse_gpus(args.gpus)
    workers_by_gpu = parse_workers_per_gpu(args.workers_per_gpu, gpus)
    requested_rows = (
        sorted(set(args.row_indices))
        if args.row_indices is not None
        else sorted(FRAMEWORKS)
    )
    unsupported_rows = set(requested_rows) - set(FRAMEWORKS)
    if unsupported_rows:
        raise ValueError(
            f"Unsupported --row-indices values: {sorted(unsupported_rows)}"
        )
    output_root = resolve_path(args.output_dir)

    total_succeeded = total_skipped = total_failed = 0
    for input_csv in input_csvs:
        output_dir = output_root / input_csv.parent.name
        succeeded, skipped, failed = predict_library(
            args,
            input_csv,
            output_dir,
            gpus,
            workers_by_gpu,
            requested_rows,
            tqdm,
        )
        total_succeeded += succeeded
        total_skipped += skipped
        total_failed += failed

    print(
        f"All libraries: completed={total_succeeded}; "
        f"skipped/resumed={total_skipped}; failed={total_failed}"
    )
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
