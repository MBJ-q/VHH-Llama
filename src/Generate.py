from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import torch
import yaml
from transformers import LlamaForCausalLM

from Prepare_data import build_default_tokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gen.yaml"
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT / "examples" / "therapeutic_vhh_scaffolds.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Outputs" / "VHH-Llama"
CHECKPOINTS_DIR_NAME = "checkpoints"
REQUIRED_INPUT_COLUMNS = {"L_seq", "R_seq"}
OUTPUT_COLUMNS = [
    "row_index",
    "sample_index",
    "generated_cdr3",
    "length",
    "eos_reached",
]


@dataclass
class GenerateArgs:
    version: str
    checkpoint: str
    minimum_sampling_batch_size: int
    max_sampling_attempts_per_input: int
    input_csv: str = str(DEFAULT_INPUT_CSV)
    sampling_batch_size: int = 128
    samples_return_per_input: int = 1
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 42
    device: list[int] = field(default_factory=lambda: [0])


@dataclass
class PromptSample:
    row_index: int
    name: str
    l_seq: str
    r_seq: str
    reference_cdr3: str | None
    input_ids: list[int]


@dataclass
class SkipCounts:
    empty_required_field: int = 0
    too_long: int = 0

    @property
    def total(self) -> int:
        return self.empty_required_field + self.too_long


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml_config(config_path: Path) -> dict:
    config_path = resolve_path(str(config_path))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_devices(value) -> list[int]:
    if isinstance(value, (list, tuple)):
        device_values = value
    else:
        device_text = str(value).strip()
        if device_text.startswith("[") and device_text.endswith("]"):
            device_text = device_text[1:-1]
        device_values = device_text.split(",")

    try:
        devices = [int(device) for device in device_values]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "generate.device must contain GPU indices such as "
            "0, 0,1, [0], or [0,1,2]"
        ) from error

    if not devices:
        raise ValueError("generate.device must contain at least one GPU index")
    if any(device < 0 for device in devices):
        raise ValueError("generate.device GPU indices must be non-negative")
    return devices


def build_args_from_config(config: dict) -> GenerateArgs:
    model_config = dict(config.get("model") or {})
    data_config = dict(config.get("data") or {})
    generate_config = dict(config.get("generate") or {})

    required_model_fields = ("version", "checkpoint")
    missing_model_fields = [
        f"model.{field}"
        for field in required_model_fields
        if not model_config.get(field)
    ]
    if missing_model_fields:
        raise ValueError(
            "Missing required config field(s): "
            + ", ".join(missing_model_fields)
        )

    required_generate_fields = (
        "minimum_sampling_batch_size",
        "max_sampling_attempts_per_input",
    )
    missing_generate_fields = [
        f"generate.{field}"
        for field in required_generate_fields
        if field not in generate_config
    ]
    if missing_generate_fields:
        raise ValueError(
            "Missing required config field(s): "
            + ", ".join(missing_generate_fields)
        )

    return GenerateArgs(
        input_csv=str(data_config.get("input_csv") or DEFAULT_INPUT_CSV),
        version=str(model_config["version"]),
        checkpoint=str(model_config["checkpoint"]),
        minimum_sampling_batch_size=int(
            generate_config["minimum_sampling_batch_size"]
        ),
        max_sampling_attempts_per_input=int(
            generate_config["max_sampling_attempts_per_input"]
        ),
        sampling_batch_size=int(
            generate_config.get("sampling_batch_size", 128)
        ),
        samples_return_per_input=int(
            generate_config.get("samples_return_per_input", 1)
        ),
        max_new_tokens=int(generate_config.get("max_new_tokens", 32)),
        temperature=float(generate_config.get("temperature", 1.0)),
        top_p=float(generate_config.get("top_p", 0.95)),
        seed=int(generate_config.get("seed", 42)),
        device=normalize_devices(generate_config.get("device", 0)),
    )


def resolve_checkpoint_path(
    checkpoint_text: str,
    version: str,
) -> Path:
    checkpoint_path = Path(checkpoint_text).expanduser()
    candidates: list[Path] = []

    if checkpoint_path.is_absolute():
        candidates.append(checkpoint_path)
    else:
        version_dir = DEFAULT_OUTPUT_ROOT / version
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
                DEFAULT_OUTPUT_ROOT / parts[0] / CHECKPOINTS_DIR_NAME / parts[1]
            )

        candidates.extend(
            [
                PROJECT_ROOT / checkpoint_path,
                DEFAULT_OUTPUT_ROOT / checkpoint_path,
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


def build_next_output_paths(
    output_dir: Path,
    version: str,
    checkpoint_path: Path,
) -> tuple[Path, Path, Path]:
    filename_prefix = f"gen_{version}_{checkpoint_path.name}"
    filename_pattern = re.compile(
        rf"^{re.escape(filename_prefix)}\((\d+)\)\.csv$"
    )
    existing_indices = []

    for path in output_dir.glob(f"{filename_prefix}(*).csv"):
        match = filename_pattern.fullmatch(path.name)
        if match:
            existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=0) + 1
    output_stem = f"{filename_prefix}({next_index})"
    return (
        output_dir / f"{output_stem}.csv",
        output_dir / f"{output_stem}_config.yaml",
        output_dir / f"{output_stem}_stats.json",
    )


def resolve_device(devices: Sequence[int]) -> torch.device:
    if len(devices) > 1:
        raise NotImplementedError(
            "Multiple GPU indices were parsed successfully, but multi-GPU "
            "generation has not been implemented yet"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    device_index = devices[0]
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_index} was requested, but only "
            f"{torch.cuda.device_count()} device(s) are visible"
        )
    return torch.device(f"cuda:{device_index}")


def validate_args(args: GenerateArgs) -> None:
    if not args.version:
        raise ValueError("model.version must not be empty")
    if not args.checkpoint:
        raise ValueError("model.checkpoint must not be empty")
    if args.sampling_batch_size <= 0:
        raise ValueError("generate.sampling_batch_size must be greater than 0")
    if args.minimum_sampling_batch_size <= 0:
        raise ValueError(
            "generate.minimum_sampling_batch_size must be greater than 0"
        )
    if args.minimum_sampling_batch_size > args.sampling_batch_size:
        raise ValueError(
            "generate.minimum_sampling_batch_size must not exceed "
            "generate.sampling_batch_size"
        )
    if args.samples_return_per_input <= 0:
        raise ValueError(
            "generate.samples_return_per_input must be greater than 0"
        )
    if args.max_sampling_attempts_per_input < args.samples_return_per_input:
        raise ValueError(
            "generate.max_sampling_attempts_per_input must be at least "
            "generate.samples_return_per_input"
        )
    if args.max_new_tokens <= 0:
        raise ValueError("generate.max_new_tokens must be greater than 0")
    if args.temperature <= 0:
        raise ValueError("generate.temperature must be greater than 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("generate.top_p must be in the interval (0, 1]")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_csv_rows(input_csv: Path) -> Iterator[tuple[int, dict[str, str]]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(REQUIRED_INPUT_COLUMNS - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(missing_columns)}"
            )

        for row_index, row in enumerate(reader):
            yield row_index, row


def encode_prompt(
    row_index: int,
    row: dict[str, str],
    tokenizer,
    max_prompt_length: int,
    skip_counts: SkipCounts,
) -> PromptSample | None:
    l_seq = (row.get("L_seq") or "").strip()
    r_seq = (row.get("R_seq") or "").strip()
    if not l_seq or not r_seq:
        skip_counts.empty_required_field += 1
        return None

    prompt_tokens = (
        list(l_seq)
        + [tokenizer.mask_token]
        + list(r_seq)
        + [tokenizer.sep_token]
    )
    input_ids = tokenizer.encode_tokens(prompt_tokens)
    if len(input_ids) > max_prompt_length:
        skip_counts.too_long += 1
        return None

    reference_cdr3 = (row.get("CDR3_seq") or "").strip() or None
    return PromptSample(
        row_index=row_index,
        name=(row.get("name") or str(row_index)).strip(),
        l_seq=l_seq,
        r_seq=r_seq,
        reference_cdr3=reference_cdr3,
        input_ids=input_ids,
    )


def iter_prompt_samples(
    input_csv: Path,
    tokenizer,
    max_prompt_length: int,
    skip_counts: SkipCounts,
) -> Iterator[PromptSample]:
    for row_index, row in iter_csv_rows(input_csv):
        sample = encode_prompt(
            row_index=row_index,
            row=row,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            skip_counts=skip_counts,
        )
        if sample is None:
            continue

        yield sample


def build_prompt_tensors(
    sample: PromptSample,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def decode_generated_ids(
    generated_ids: Sequence[int],
    tokenizer,
) -> tuple[str, bool]:
    amino_acids: list[str] = []
    eos_reached = False

    for token_id in generated_ids:
        if token_id == tokenizer.eos_token_id:
            eos_reached = True
            break
        amino_acids.append(tokenizer.id_to_token[token_id])

    return "".join(amino_acids), eos_reached


@torch.inference_mode()
def generate_sample_batch(
    model: LlamaForCausalLM,
    sample: PromptSample,
    tokenizer,
    device: torch.device,
    args: GenerateArgs,
    num_samples: int,
) -> list[dict]:
    input_ids, attention_mask = build_prompt_tensors(
        sample=sample,
        device=device,
    )
    prompt_width = input_ids.size(1)
    blocked_token_ids = [
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.mask_token_id,
        tokenizer.sep_token_id,
    ]

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        suppress_tokens=blocked_token_ids,
        use_cache=True,
    )
    generated = generated[:, prompt_width:].detach().cpu().tolist()

    generated_samples = []
    for token_ids in generated:
        generated_cdr3, eos_reached = decode_generated_ids(token_ids, tokenizer)
        generated_samples.append(
            {
                "generated_cdr3": generated_cdr3,
                "length": len(generated_cdr3),
                "eos_reached": eos_reached,
            }
        )

    return generated_samples


def make_read_only(path: Path) -> None:
    path.chmod(0o444)


def build_prompt_progress_bar(prompt_sample: PromptSample, total: int):
    if tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=f"Framework row {prompt_sample.row_index}",
        unit="unique sequences",
        dynamic_ncols=True,
        leave=True,
    )


def update_prompt_progress_bar(
    progress_bar,
    completed_unique: int,
    unique: int,
    attempts: int,
    duplicates: int,
) -> None:
    if progress_bar is None:
        return
    progress_bar.update(completed_unique)
    progress_bar.set_postfix(
        unique=unique,
        attempts=attempts,
        duplicates=duplicates,
    )


def generate(args: GenerateArgs, config_path: Path) -> None:
    validate_args(args)
    input_csv = resolve_path(args.input_csv)
    output_dir = input_csv.parent
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_text=args.checkpoint,
        version=args.version,
    )
    output_csv, output_config_yaml, output_stats_json = build_next_output_paths(
        output_dir,
        args.version,
        checkpoint_path,
    )
    partial_output_csv = output_csv.with_suffix(f"{output_csv.suffix}.part")
    partial_config_yaml = output_config_yaml.with_suffix(
        f"{output_config_yaml.suffix}.part"
    )
    partial_stats_json = output_stats_json.with_suffix(
        f"{output_stats_json.suffix}.part"
    )

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    device = resolve_device(args.device)
    set_seed(args.seed)
    tokenizer = build_default_tokenizer()
    model = LlamaForCausalLM.from_pretrained(str(checkpoint_path))
    model.to(device)
    model.eval()
    model.config.use_cache = True

    model_max_length = int(model.config.max_position_embeddings)
    max_prompt_length = model_max_length - args.max_new_tokens
    if max_prompt_length <= 0:
        raise ValueError(
            "generate.max_new_tokens must be smaller than the model's "
            f"max_position_embeddings ({model_max_length})"
        )

    skip_counts = SkipCounts()
    prompt_samples = iter_prompt_samples(
        input_csv=input_csv,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        skip_counts=skip_counts,
    )

    started_at = time.monotonic()
    processed_inputs = 0
    sampling_attempts = 0
    unique_sequences = 0
    duplicate_sequences_discarded = 0
    framework_stats: list[dict] = []
    failure_message: str | None = None

    shutil.copy2(config_path, partial_config_yaml)
    with partial_output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for prompt_sample in prompt_samples:
            attempts_for_prompt = 0
            unique_for_prompt = 0
            duplicates_for_prompt = 0
            seen_sequences: set[str] = set()
            progress_bar = build_prompt_progress_bar(
                prompt_sample=prompt_sample,
                total=args.samples_return_per_input,
            )

            try:
                while (
                    unique_for_prompt < args.samples_return_per_input
                    and attempts_for_prompt
                    < args.max_sampling_attempts_per_input
                ):
                    remaining_unique = (
                        args.samples_return_per_input - unique_for_prompt
                    )
                    remaining_attempts = (
                        args.max_sampling_attempts_per_input
                        - attempts_for_prompt
                    )
                    current_batch_size = min(
                        args.sampling_batch_size,
                        max(
                            remaining_unique,
                            args.minimum_sampling_batch_size,
                        ),
                        remaining_attempts,
                    )
                    sample_records = generate_sample_batch(
                        model=model,
                        sample=prompt_sample,
                        tokenizer=tokenizer,
                        device=device,
                        args=args,
                        num_samples=current_batch_size,
                    )
                    attempts_for_prompt += len(sample_records)
                    sampling_attempts += len(sample_records)

                    unique_added = 0
                    for sample_record in sample_records:
                        if unique_for_prompt >= args.samples_return_per_input:
                            break
                        generated_cdr3 = sample_record["generated_cdr3"]
                        if generated_cdr3 in seen_sequences:
                            duplicates_for_prompt += 1
                            duplicate_sequences_discarded += 1
                            continue

                        seen_sequences.add(generated_cdr3)
                        writer.writerow(
                            {
                                "row_index": prompt_sample.row_index,
                                "sample_index": unique_for_prompt,
                                **sample_record,
                            }
                        )
                        unique_for_prompt += 1
                        unique_sequences += 1
                        unique_added += 1

                    update_prompt_progress_bar(
                        progress_bar=progress_bar,
                        completed_unique=unique_added,
                        unique=unique_for_prompt,
                        attempts=attempts_for_prompt,
                        duplicates=duplicates_for_prompt,
                    )
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            del seen_sequences
            completed = unique_for_prompt >= args.samples_return_per_input
            framework_stats.append(
                {
                    "row_index": prompt_sample.row_index,
                    "name": prompt_sample.name,
                    "target_unique": args.samples_return_per_input,
                    "unique_generated": unique_for_prompt,
                    "raw_generated_count": attempts_for_prompt,
                    "duplicate_count": duplicates_for_prompt,
                    "completed": completed,
                }
            )
            processed_inputs += 1
            if not completed:
                failure_message = (
                    f"Framework row {prompt_sample.row_index} reached the "
                    "sampling limit: "
                    f"target={args.samples_return_per_input}, "
                    f"unique={unique_for_prompt}, "
                    f"attempts={attempts_for_prompt}, "
                    f"duplicates={duplicates_for_prompt}"
                )
                break

    elapsed = time.monotonic() - started_at
    with partial_stats_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_csv": str(input_csv),
                "samples_return_per_input": args.samples_return_per_input,
                "frameworks": framework_stats,
                "total_raw_generated_count": sampling_attempts,
                "total_unique_generated": unique_sequences,
                "total_duplicate_count": duplicate_sequences_discarded,
                "completed": failure_message is None,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    if failure_message is not None:
        raise RuntimeError(failure_message)

    partial_output_csv.replace(output_csv)
    partial_config_yaml.replace(output_config_yaml)
    partial_stats_json.replace(output_stats_json)
    make_read_only(output_csv)
    make_read_only(output_config_yaml)
    make_read_only(output_stats_json)

    print(f"Input CSV: {input_csv}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Config snapshot: {output_config_yaml}")
    print(f"Sampling stats: {output_stats_json}")
    print(f"Device: {device}")
    print(f"Processed inputs: {processed_inputs:,}")
    print(f"Sampling attempts: {sampling_attempts:,}")
    print(f"Unique sequences: {unique_sequences:,}")
    print(f"Duplicate sequences discarded: {duplicate_sequences_discarded:,}")
    print(f"Skipped empty required field: {skip_counts.empty_required_field:,}")
    print(f"Skipped too long: {skip_counts.too_long:,}")
    print(f"Elapsed seconds: {elapsed:.1f}")


def parse_args() -> tuple[GenerateArgs, Path]:
    parser = argparse.ArgumentParser(
        description="Sample CDR3 sequences from a VHH-Llama checkpoint."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    cli_args = parser.parse_args()
    config_path = resolve_path(str(cli_args.config))
    generate_args = build_args_from_config(load_yaml_config(config_path))
    return generate_args, config_path


def main() -> None:
    generate_args, config_path = parse_args()
    generate(args=generate_args, config_path=config_path)


if __name__ == "__main__":
    main()
