from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from Prepare_data import build_default_tokenizer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_INPUT_CSV = PROJECT_ROOT / "datasets" / "test.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "VHH-Llama"
DEFAULT_PPL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "PPL"
DEFAULT_MAX_LENGTH = 170
CHECKPOINTS_DIR_NAME = "checkpoints"
SPECIAL_META_COLUMNS = ("subject_species", "subject_name", "subject_sex")
PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
)

OUTPUT_COLUMNS = [
    "row_index",
    "subject_species",
    "subject_name",
    "subject_sex",
    "CDR3_seq",
    "cdr3_length",
    "cdr3_position",
    "cdr3_aa",
    "token_probability",
    "site_nll",
    "site_ppl",
    "cdr3_residue_mean_nll",
    "cdr3_residue_ppl",
    "cdr3_residue&eos_mean_nll",
    "cdr3_residue&eos_ppl",
]


@dataclass
class EncodedSample:
    row_index: int
    meta: dict[str, str]
    cdr3_seq: str
    cdr3_length: int
    prompt_length: int
    input_ids: list[int]


@dataclass
class SkipCounts:
    empty_required_field: int = 0
    too_long: int = 0

    @property
    def total(self) -> int:
        return self.empty_required_field + self.too_long


@dataclass
class ShardProcess:
    shard_index: int
    gpu_id: str
    row_start: int
    row_end: int
    process: subprocess.Popen
    log_path: Path
    progress_path: Path


def require_tqdm() -> None:
    if tqdm is None:
        raise RuntimeError(
            "tqdm is required for PPL progress bars. "
            "Install it in the active environment with: python -m pip install tqdm"
        )


def open_progress_stream():
    try:
        return Path("/dev/tty").open("w")
    except OSError:
        return sys.stderr


def close_progress_stream(progress_stream) -> None:
    if progress_stream not in (sys.stderr, sys.stdout):
        progress_stream.close()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_checkpoint_path(
    checkpoint_text: str,
    version: str | None,
    output_root: Path,
) -> Path:
    checkpoint_path = Path(checkpoint_text).expanduser()
    candidates = []

    if checkpoint_path.is_absolute():
        candidates.append(checkpoint_path)
        if (
            checkpoint_path.name.startswith("checkpoint-")
            and checkpoint_path.parent.name.startswith("version")
        ):
            candidates.append(
                checkpoint_path.parent / CHECKPOINTS_DIR_NAME / checkpoint_path.name
            )
    else:
        if version:
            version_dir = output_root / version
            candidates.append(version_dir / checkpoint_path)
            if checkpoint_path.name.startswith("checkpoint-"):
                candidates.append(
                    version_dir / CHECKPOINTS_DIR_NAME / checkpoint_path.name
                )

        parts = checkpoint_path.parts
        if (
            len(parts) == 2
            and parts[0].startswith("version")
            and parts[1].startswith("checkpoint-")
        ):
            candidates.append(
                output_root / parts[0] / CHECKPOINTS_DIR_NAME / parts[1]
            )

        candidates.extend(
            [
                PROJECT_ROOT / checkpoint_path,
                output_root / checkpoint_path,
            ]
        )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            if not resolved.is_dir():
                raise ValueError(f"Checkpoint path must be a directory: {resolved}")
            return resolved

    searched = "\n".join(f"  - {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}")


def infer_version_name(model_path: Path, version: str | None) -> str | None:
    if version:
        return version
    for part in model_path.parts:
        if part.startswith("version"):
            return part
    return None


def build_default_output_path(
    input_csv: Path,
    checkpoint_path: Path,
    version: str | None,
) -> Path:
    version_name = infer_version_name(checkpoint_path, version)
    model_label = (
        f"{version_name}_{checkpoint_path.name}"
        if version_name
        else checkpoint_path.name
    )
    return DEFAULT_PPL_OUTPUT_DIR / f"ppl_{model_label}.csv"


def resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def iter_csv_rows(input_csv: Path) -> Iterator[tuple[int, dict[str, str]]]:
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"L_seq", "CDR3_seq", "R_seq"}
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(missing_columns)}"
            )

        for row_index, row in enumerate(reader):
            yield row_index, row


def encode_sample(
    row_index: int,
    row: dict[str, str],
    tokenizer,
    max_length: int,
    skip_counts: SkipCounts,
) -> EncodedSample | None:
    l_seq = (row.get("L_seq") or "").strip()
    cdr3_seq = (row.get("CDR3_seq") or "").strip()
    r_seq = (row.get("R_seq") or "").strip()
    if not l_seq or not cdr3_seq or not r_seq:
        skip_counts.empty_required_field += 1
        return None

    prompt_tokens = (
        list(l_seq)
        + [tokenizer.mask_token]
        + list(r_seq)
        + [tokenizer.sep_token]
    )
    target_tokens = list(cdr3_seq) + [tokenizer.eos_token]
    tokens = prompt_tokens + target_tokens
    if len(tokens) > max_length:
        skip_counts.too_long += 1
        return None

    return EncodedSample(
        row_index=row_index,
        meta={column: (row.get(column) or "") for column in SPECIAL_META_COLUMNS},
        cdr3_seq=cdr3_seq,
        cdr3_length=len(cdr3_seq),
        prompt_length=len(prompt_tokens),
        input_ids=tokenizer.encode_tokens(tokens),
    )


def batched(
    iterator: Iterator[EncodedSample],
    batch_size: int,
) -> Iterator[list[EncodedSample]]:
    batch = []
    for sample in iterator:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_encoded_samples(
    input_csv: Path,
    tokenizer,
    max_length: int,
    skip_counts: SkipCounts,
    limit: int | None,
    row_start: int = 0,
    row_end: int | None = None,
) -> Iterator[EncodedSample]:
    yielded = 0
    for row_index, row in iter_csv_rows(input_csv):
        if row_index < row_start:
            continue
        if row_end is not None and row_index >= row_end:
            return

        sample = encode_sample(
            row_index=row_index,
            row=row,
            tokenizer=tokenizer,
            max_length=max_length,
            skip_counts=skip_counts,
        )
        if sample is None:
            continue

        yield sample
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def pad_batch(
    samples: Sequence[EncodedSample],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(sample.input_ids) for sample in samples)
    input_ids = []
    attention_mask = []

    for sample in samples:
        pad_length = max_length - len(sample.input_ids)
        input_ids.append(sample.input_ids + [pad_token_id] * pad_length)
        attention_mask.append([1] * len(sample.input_ids) + [0] * pad_length)

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
    )


def format_float(value: float) -> str:
    return f"{value:.10g}"


def build_output_rows(
    sample: EncodedSample,
    token_probabilities: list[float],
    site_nll_values: list[float],
) -> list[dict[str, str | int]]:
    residue_nll_values = site_nll_values[: sample.cdr3_length]
    residue_mean_nll = sum(residue_nll_values) / len(residue_nll_values)
    residue_ppl = math.exp(residue_mean_nll)
    residue_eos_mean_nll = sum(site_nll_values) / len(site_nll_values)
    residue_eos_ppl = math.exp(residue_eos_mean_nll)

    rows = []
    target_tokens = list(sample.cdr3_seq) + ["[EOS]"]
    for target_index, target_token in enumerate(target_tokens):
        site_nll = site_nll_values[target_index]
        site_ppl = math.exp(site_nll)
        rows.append(
            {
                "row_index": sample.row_index,
                "subject_species": sample.meta["subject_species"],
                "subject_name": sample.meta["subject_name"],
                "subject_sex": sample.meta["subject_sex"],
                "CDR3_seq": sample.cdr3_seq,
                "cdr3_length": sample.cdr3_length,
                "cdr3_position": target_index + 1,
                "cdr3_aa": target_token,
                "token_probability": format_float(token_probabilities[target_index]),
                "site_nll": format_float(site_nll),
                "site_ppl": format_float(site_ppl),
                "cdr3_residue_mean_nll": format_float(residue_mean_nll),
                "cdr3_residue_ppl": format_float(residue_ppl),
                "cdr3_residue&eos_mean_nll": format_float(residue_eos_mean_nll),
                "cdr3_residue&eos_ppl": format_float(residue_eos_ppl),
            }
        )
    return rows


@torch.inference_mode()
def write_batch_ppl(
    model: LlamaForCausalLM,
    samples: Sequence[EncodedSample],
    tokenizer,
    device: torch.device,
    writer: csv.DictWriter,
) -> int:
    input_ids, attention_mask = pad_batch(
        samples=samples,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    written_rows = 0
    for batch_index, sample in enumerate(samples):
        target_start = sample.prompt_length
        target_end = sample.prompt_length + sample.cdr3_length + 1
        target_positions = torch.arange(
            target_start,
            target_end,
            device=device,
            dtype=torch.long,
        )
        target_ids = input_ids[batch_index, target_positions]
        next_token_log_probs = shifted_log_probs[
            batch_index,
            target_positions - 1,
            target_ids,
        ]
        site_nll_values = (-next_token_log_probs).detach().cpu().tolist()
        token_probabilities = next_token_log_probs.exp().detach().cpu().tolist()

        for output_row in build_output_rows(
            sample=sample,
            token_probabilities=token_probabilities,
            site_nll_values=site_nll_values,
        ):
            writer.writerow(output_row)
            written_rows += 1

    return written_rows


def resolve_evaluation_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    input_csv = resolve_path(args.input_csv)
    output_root = resolve_path(args.output_root)
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_text=args.checkpoint,
        version=args.version,
        output_root=output_root,
    )
    output_csv = (
        resolve_path(args.output_csv)
        if args.output_csv
        else build_default_output_path(input_csv, checkpoint_path, args.version)
    )
    return input_csv, output_root, checkpoint_path, output_csv


def validate_row_range(row_start: int, row_end: int | None) -> None:
    if row_start < 0:
        raise ValueError(f"row_start must be >= 0, got {row_start}")
    if row_end is not None and row_end < row_start:
        raise ValueError(f"row_end must be >= row_start, got {row_end} < {row_start}")


def evaluate(args: argparse.Namespace) -> None:
    input_csv, _output_root, checkpoint_path, output_csv = resolve_evaluation_paths(args)
    validate_row_range(args.worker_row_start, args.worker_row_end)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output CSV already exists: {output_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    tokenizer = build_default_tokenizer()

    model = LlamaForCausalLM.from_pretrained(str(checkpoint_path))
    model.config.use_cache = False
    model.to(device)
    model.eval()

    max_length = int(args.max_length)
    skip_counts = SkipCounts()
    encoded_samples = iter_encoded_samples(
        input_csv=input_csv,
        tokenizer=tokenizer,
        max_length=max_length,
        skip_counts=skip_counts,
        limit=args.limit,
        row_start=args.worker_row_start,
        row_end=args.worker_row_end,
    )

    progress_total = resolve_progress_total(args, input_csv)
    progress_completed = 0
    progress_bar = None
    progress_stream = None
    last_progress_update = 0.0
    if args.progress_file:
        write_progress_file(resolve_path(args.progress_file), 0, progress_total)
    started_at = time.monotonic()
    processed_samples = 0
    written_rows = 0
    if not args.progress_file and not args.no_progress_bar:
        require_tqdm()
        progress_stream = open_progress_stream()
        progress_bar = tqdm(
            total=progress_total,
            desc="PPL",
            unit="rows",
            ncols=args.progress_ncols,
            bar_format=PROGRESS_BAR_FORMAT,
            file=progress_stream,
            mininterval=args.progress_interval,
            leave=True,
        )

    try:
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            for batch in batched(encoded_samples, args.batch_size):
                written_rows += write_batch_ppl(
                    model=model,
                    samples=batch,
                    tokenizer=tokenizer,
                    device=device,
                    writer=writer,
                )
                processed_samples += len(batch)
                if args.limit is not None and args.worker_row_end is None:
                    progress_completed = processed_samples
                else:
                    progress_completed = progress_from_batch(args, batch)

                if progress_bar is not None:
                    delta = progress_completed - progress_bar.n
                    if delta > 0:
                        progress_bar.update(delta)

                now = time.monotonic()
                if args.progress_file and now - last_progress_update >= args.progress_interval:
                    write_progress_file(
                        resolve_path(args.progress_file),
                        progress_completed,
                        progress_total,
                    )
                    last_progress_update = now
    finally:
        if progress_bar is not None:
            delta = progress_total - progress_bar.n
            if delta > 0:
                progress_bar.update(delta)
            progress_bar.close()
        if progress_stream is not None:
            close_progress_stream(progress_stream)

    elapsed = time.monotonic() - started_at
    if args.progress_file:
        write_progress_file(
            resolve_path(args.progress_file),
            progress_total,
            progress_total,
        )

    print(f"Input CSV: {input_csv}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Device: {device}")
    print(f"Max length: {max_length}")
    print(f"Worker row start: {args.worker_row_start}")
    print(f"Worker row end: {args.worker_row_end}")
    print(f"Processed samples: {processed_samples:,}")
    print(f"Written rows: {written_rows:,}")
    print(f"Skipped empty required field: {skip_counts.empty_required_field:,}")
    print(f"Skipped too long: {skip_counts.too_long:,}")
    print(f"Elapsed seconds: {elapsed:.1f}")


def parse_visible_gpus(visible_gpus: str) -> list[str]:
    gpu_ids = [item.strip() for item in visible_gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--visible_gpus must contain at least one GPU id")
    return gpu_ids


def count_csv_data_rows(input_csv: Path) -> int:
    with input_csv.open(newline="") as handle:
        row_count = sum(1 for _line in handle)
    return max(0, row_count - 1)


def write_progress_file(progress_path: Path, completed: int, total: int) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_name(f"{progress_path.name}.tmp")
    tmp_path.write_text(f"{completed},{total},{time.time()}\n")
    tmp_path.replace(progress_path)


def read_progress_file(progress_path: Path) -> int:
    try:
        text = progress_path.read_text().strip()
        if not text:
            return 0
        return int(text.split(",", 1)[0])
    except (FileNotFoundError, OSError, ValueError):
        return 0


def resolve_progress_total(args: argparse.Namespace, input_csv: Path) -> int:
    if args.worker_row_end is not None:
        return max(0, args.worker_row_end - args.worker_row_start)
    if args.limit is not None:
        return int(args.limit)
    return count_csv_data_rows(input_csv)


def progress_from_batch(args: argparse.Namespace, batch: Sequence[EncodedSample]) -> int:
    if args.worker_row_end is not None:
        return max(
            0,
            min(args.worker_row_end, batch[-1].row_index + 1)
            - args.worker_row_start,
        )
    if args.limit is not None:
        return 0
    return batch[-1].row_index + 1


def build_row_ranges(total_rows: int, num_shards: int) -> list[tuple[int, int]]:
    base_size = total_rows // num_shards
    remainder = total_rows % num_shards
    ranges = []
    row_start = 0
    for shard_index in range(num_shards):
        shard_size = base_size + (1 if shard_index < remainder else 0)
        row_end = row_start + shard_size
        ranges.append((row_start, row_end))
        row_start = row_end
    return ranges


def build_shard_path(output_csv: Path, shard_index: int, num_shards: int) -> Path:
    return output_csv.with_name(
        f"{output_csv.stem}.shard{shard_index}of{num_shards}{output_csv.suffix}"
    )


def build_shard_log_path(output_csv: Path, shard_index: int, num_shards: int) -> Path:
    return output_csv.with_name(
        f"{output_csv.stem}.shard{shard_index}of{num_shards}.log"
    )


def build_shard_progress_path(
    output_csv: Path,
    shard_index: int,
    num_shards: int,
) -> Path:
    return output_csv.with_name(
        f"{output_csv.stem}.shard{shard_index}of{num_shards}.progress"
    )


def build_worker_command(
    args: argparse.Namespace,
    checkpoint_path: Path,
    shard_output_csv: Path,
    progress_path: Path,
    row_start: int,
    row_end: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input_csv",
        str(resolve_path(args.input_csv)),
        "--checkpoint",
        str(checkpoint_path),
        "--output_root",
        str(resolve_path(args.output_root)),
        "--output_csv",
        str(shard_output_csv),
        "--batch_size",
        str(args.batch_size),
        "--device",
        "cuda:0",
        "--worker_row_start",
        str(row_start),
        "--worker_row_end",
        str(row_end),
        "--log_every",
        str(args.log_every),
        "--progress_file",
        str(progress_path),
        "--progress_interval",
        str(args.progress_interval),
        "--no_progress_bar",
        "--overwrite",
    ]
    if args.version:
        command.extend(["--version", args.version])
    if args.max_length is not None:
        command.extend(["--max_length", str(args.max_length)])
    return command


def merge_shard_csvs(shard_paths: Sequence[Path], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with output_csv.open("w", newline="") as output_handle:
        for shard_path in shard_paths:
            with shard_path.open(newline="") as shard_handle:
                header = shard_handle.readline()
                if not header:
                    continue
                if not wrote_header:
                    output_handle.write(header)
                    wrote_header = True
                for line in shard_handle:
                    output_handle.write(line)


def remove_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def run_visible_gpu_shards(args: argparse.Namespace) -> None:
    gpu_ids = parse_visible_gpus(args.visible_gpus)
    input_csv, _output_root, checkpoint_path, output_csv = resolve_evaluation_paths(args)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output CSV already exists: {output_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_rows = count_csv_data_rows(input_csv)
    if args.limit is not None:
        total_rows = min(total_rows, int(args.limit))

    num_shards = len(gpu_ids)
    row_ranges = build_row_ranges(total_rows, num_shards)
    shard_paths = [
        build_shard_path(output_csv, shard_index, num_shards)
        for shard_index in range(num_shards)
    ]
    log_paths = [
        build_shard_log_path(output_csv, shard_index, num_shards)
        for shard_index in range(num_shards)
    ]
    progress_paths = [
        build_shard_progress_path(output_csv, shard_index, num_shards)
        for shard_index in range(num_shards)
    ]
    remove_paths(progress_paths)

    started_at = time.monotonic()
    shards = []
    log_handles = []
    for shard_index, (
        gpu_id,
        row_range,
        shard_path,
        log_path,
        progress_path,
    ) in enumerate(
        zip(gpu_ids, row_ranges, shard_paths, log_paths, progress_paths)
    ):
        row_start, row_end = row_range
        command = build_worker_command(
            args=args,
            checkpoint_path=checkpoint_path,
            shard_output_csv=shard_path,
            progress_path=progress_path,
            row_start=row_start,
            row_end=row_end,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        log_handle = log_path.open("w")
        log_handles.append(log_handle)
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        shards.append(
            ShardProcess(
                shard_index=shard_index,
                gpu_id=gpu_id,
                row_start=row_start,
                row_end=row_end,
                process=process,
                log_path=log_path,
                progress_path=progress_path,
            )
        )

    failed = []
    require_tqdm()
    progress_stream = open_progress_stream()
    progress_bar = tqdm(
        total=total_rows,
        desc="PPL total",
        unit="rows",
        ncols=args.progress_ncols,
        bar_format=PROGRESS_BAR_FORMAT,
        file=progress_stream,
        mininterval=args.progress_interval,
        leave=True,
    )
    try:
        while True:
            completed_rows = 0
            done_shards = 0
            running_shards = 0

            for shard in shards:
                shard_total = shard.row_end - shard.row_start
                return_code = shard.process.poll()
                if return_code is None:
                    running_shards += 1
                    completed_rows += min(
                        read_progress_file(shard.progress_path),
                        shard_total,
                    )
                    continue

                done_shards += 1
                if return_code == 0:
                    completed_rows += shard_total
                else:
                    completed_rows += min(
                        read_progress_file(shard.progress_path),
                        shard_total,
                    )

            delta = completed_rows - progress_bar.n
            if delta > 0:
                progress_bar.update(delta)
            if running_shards == 0:
                break
            time.sleep(args.progress_interval)
    finally:
        progress_bar.close()
        close_progress_stream(progress_stream)
        for log_handle in log_handles:
            log_handle.close()

    for shard in shards:
        return_code = shard.process.returncode
        if return_code != 0:
            failed.append(
                (shard.shard_index, shard.gpu_id, return_code, shard.log_path)
            )

    if failed:
        details = "\n".join(
            f"  shard {shard_index} on GPU {gpu_id} failed with code "
            f"{return_code}; log: {log_path}"
            for shard_index, gpu_id, return_code, log_path in failed
        )
        raise RuntimeError(f"One or more PPL shards failed:\n{details}")

    merge_shard_csvs(shard_paths, output_csv)
    if not args.keep_shards:
        remove_paths([*shard_paths, *log_paths, *progress_paths])

    elapsed = time.monotonic() - started_at
    print(f"Input CSV: {input_csv}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Visible GPUs: {','.join(gpu_ids)}")
    print(f"Shards: {num_shards}")
    print(f"Elapsed seconds: {elapsed:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-site CDR3 perplexity for a VHH-Llama checkpoint."
    )
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--version", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10_000)
    parser.add_argument("--visible_gpus", default=None)
    parser.add_argument(
        "--worker_row_start",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker_row_end",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--progress_interval", type=float, default=5.0)
    parser.add_argument("--progress_ncols", type=int, default=80)
    parser.add_argument("--progress_file", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--keep_shards", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.visible_gpus:
        run_visible_gpu_shards(args)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
