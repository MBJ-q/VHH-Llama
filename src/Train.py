from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import yaml
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.modeling_utils import unwrap_model

from Model import ModelArgs, build_model, count_parameters
from Prepare_data import (
    CDR3SubsampleSampler,
    DataArgs,
    EpochRandomSampler,
    TASK_CDR3,
    TASK_VHH,
    VHHIndexedDataset,
    build_default_tokenizer,
    collate_fn,
    count_indexed_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"
DEFAULT_TRAIN_CSV = (
    PROJECT_ROOT / "datasets" / "AbNativ2_Re_Plus" / "train.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Outputs" / "VHH-Llama"
CHECKPOINTS_DIR_NAME = "checkpoints"
FALSE_VALUES = {None, False, "", "false", "False", "none", "None", "null", "Null"}


@dataclass
class BaseTrainArgs:
    train_csv: str = str(DEFAULT_TRAIN_CSV)
    val_csv: Optional[str] = None
    output_dir: str = str(DEFAULT_OUTPUT_ROOT)
    eval_strategy: str = "no"
    eval_every_epoch: float = 1.0
    early_stopping_patience: int = 3
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 100
    save_strategy: str = "epoch"
    save_every_epoch: float = 1.0
    save_total_limit: int = 3
    max_length: int = 256
    num_workers: int = 0
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    device: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None


@dataclass
class LlamaTrainModelArgs:
    num_hidden_layers: int
    hidden_size: int
    name: str = "VHH-Llama"
    max_position_embeddings: int = 256
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0


def validate_model_args(model_args: LlamaTrainModelArgs) -> None:
    if model_args.num_attention_heads <= 0:
        raise ValueError("model.num_attention_heads must be greater than 0")
    if model_args.num_key_value_heads <= 0:
        raise ValueError("model.num_key_value_heads must be greater than 0")
    if model_args.hidden_size % model_args.num_attention_heads != 0:
        raise ValueError(
            "model.hidden_size must be divisible by "
            "model.num_attention_heads"
        )
    if model_args.num_attention_heads % model_args.num_key_value_heads != 0:
        raise ValueError(
            "model.num_attention_heads must be divisible by "
            "model.num_key_value_heads"
        )


@dataclass
class ResolvedSchedule:
    train_samples: int
    eval_samples: int
    device_count: int
    global_batch_size: int
    steps_per_epoch: int
    max_steps: int
    save_strategy: str
    save_steps: Optional[int]
    eval_strategy: str
    eval_steps: Optional[int]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_optional_path(value) -> Optional[str]:
    if value in FALSE_VALUES:
        return None
    return str(value)


def load_yaml_config(config_path: Path) -> dict:
    config_path = resolve_path(str(config_path))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_device(device) -> Optional[str]:
    if isinstance(device, Sequence) and not isinstance(device, str):
        return ",".join(str(item) for item in device)
    if device in FALSE_VALUES:
        return None
    return str(device)


def _build_base_args_from_config(
    config: dict,
) -> tuple[BaseTrainArgs, LlamaTrainModelArgs]:
    data_config = dict(config.get("data") or {})
    train_config = dict(config.get("train") or {})
    validation_config = dict(config.get("validation") or {})
    model_config = dict(config.get("model") or {})

    required_model_fields = ("hidden_size", "num_hidden_layers")
    missing_model_fields = [
        f"model.{field}"
        for field in required_model_fields
        if field not in model_config
    ]
    if missing_model_fields:
        raise ValueError(
            "Missing required config field(s): "
            + ", ".join(missing_model_fields)
        )

    train_args = BaseTrainArgs(
        train_csv=str(data_config.get("train_csv") or DEFAULT_TRAIN_CSV),
        val_csv=data_config.get("val_csv") or None,
        output_dir=str(train_config.get("output_dir") or DEFAULT_OUTPUT_ROOT),
        eval_strategy=str(validation_config.get("eval_strategy", "no")),
        eval_every_epoch=float(validation_config.get("eval_every_epoch", 1.0)),
        early_stopping_patience=int(
            validation_config.get("early_stopping_patience", 3)
        ),
        batch_size=int(train_config.get("batch_size", 32)),
        gradient_accumulation_steps=int(
            train_config.get("gradient_accumulation_steps", 1)
        ),
        learning_rate=float(train_config.get("learning_rate", 3e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.03)),
        logging_steps=int(train_config.get("logging_steps", 100)),
        save_strategy=str(train_config.get("save_strategy", "epoch")),
        save_every_epoch=float(train_config.get("save_every_epoch", 1.0)),
        save_total_limit=int(train_config.get("save_total_limit", 3)),
        max_length=int(data_config.get("max_length", 256)),
        num_workers=int(train_config.get("num_workers", 0)),
        seed=int(train_config.get("seed", 42)),
        fp16=bool(train_config.get("fp16", False)),
        bf16=bool(train_config.get("bf16", True)),
        device=normalize_device(train_config.get("device")),
        resume_from_checkpoint=normalize_optional_path(
            train_config.get("resume_from_checkpoint")
        ),
    )
    model_args = LlamaTrainModelArgs(
        name=str(model_config.get("name", "VHH-Llama")),
        max_position_embeddings=int(model_config.get("max_position_embeddings", 256)),
        num_hidden_layers=int(model_config["num_hidden_layers"]),
        hidden_size=int(model_config["hidden_size"]),
        intermediate_size=int(model_config.get("intermediate_size", 2048)),
        num_attention_heads=int(model_config.get("num_attention_heads", 8)),
        num_key_value_heads=int(model_config.get("num_key_value_heads", 8)),
        attention_dropout=float(model_config.get("attention_dropout", 0.0)),
        rms_norm_eps=float(model_config.get("rms_norm_eps", 1e-6)),
        rope_theta=float(model_config.get("rope_theta", 10000.0)),
    )
    validate_model_args(model_args)
    return train_args, model_args


@dataclass
class TrainArgs(BaseTrainArgs):
    max_steps: int = 0
    mixed_infill_enabled: bool = False
    vhh_span_probability: float = 0.0
    min_span_length: int = 5
    max_span_length: int = 15
    task_seed: int = 1
    span_length_seed: int = 2
    span_position_seed: int = 3
    cdr3_sampling_enabled: bool = False
    cdr3_full_keep_threshold: int = 100
    cdr3_sampling_power: float = 0.3
    cdr3_sampling_seed: int = 10_000_061


def validate_mixed_infill_args(train_args: TrainArgs) -> None:
    if not 0.0 <= train_args.vhh_span_probability <= 1.0:
        raise ValueError("mixed_infill.vhh_span_probability must be between 0 and 1")
    if train_args.min_span_length <= 0:
        raise ValueError("mixed_infill.min_span_length must be greater than 0")
    if train_args.min_span_length > train_args.max_span_length:
        raise ValueError(
            "mixed_infill.min_span_length must not exceed max_span_length"
        )


def validate_cdr3_sampling_args(train_args: TrainArgs) -> None:
    if train_args.cdr3_full_keep_threshold < 1:
        raise ValueError("cdr3_sampling.full_keep_threshold must be at least 1")
    if not 0.0 < train_args.cdr3_sampling_power <= 1.0:
        raise ValueError("cdr3_sampling.sampling_power must be in (0, 1]")


def build_args_from_config(
    config: dict,
) -> tuple[TrainArgs, LlamaTrainModelArgs]:
    original_train_args, model_args = _build_base_args_from_config(config)
    train_config = dict(config.get("train") or {})
    mixed_config = dict(config.get("mixed_infill") or {})
    enabled = bool(mixed_config.get("enabled", False))
    sampling_config = dict(config.get("cdr3_sampling") or {})
    sampling_enabled = bool(sampling_config.get("enabled", False))

    train_args = TrainArgs(
        **asdict(original_train_args),
        max_steps=int(train_config.get("max_steps", 0)),
        mixed_infill_enabled=enabled,
        vhh_span_probability=(
            float(mixed_config.get("vhh_span_probability", 0.2))
            if enabled
            else 0.0
        ),
        min_span_length=(
            int(mixed_config.get("min_span_length", 5)) if enabled else 5
        ),
        max_span_length=(
            int(mixed_config.get("max_span_length", 15)) if enabled else 15
        ),
        task_seed=(int(mixed_config.get("task_seed", 1)) if enabled else 1),
        span_length_seed=(
            int(mixed_config.get("span_length_seed", 2)) if enabled else 2
        ),
        span_position_seed=(
            int(mixed_config.get("span_position_seed", 3)) if enabled else 3
        ),
        cdr3_sampling_enabled=sampling_enabled,
        cdr3_full_keep_threshold=int(
            sampling_config.get("full_keep_threshold", 100)
        ),
        cdr3_sampling_power=float(sampling_config.get("sampling_power", 0.3)),
        cdr3_sampling_seed=int(
            sampling_config.get("seed", original_train_args.seed + 10_000_019)
        ),
    )
    if enabled:
        validate_mixed_infill_args(train_args)
    if sampling_enabled:
        validate_cdr3_sampling_args(train_args)
    if train_args.max_steps <= 0:
        raise ValueError("train.max_steps must be greater than 0")
    return train_args, model_args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def configure_cuda_devices(device: Optional[str]) -> None:
    if device:
        device_ids = [
            item.strip()
            for item in str(device).split(",")
            if item.strip()
        ]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
        visible_device_count = torch.cuda.device_count()
        if visible_device_count != len(device_ids):
            raise RuntimeError(
                f"Configured {len(device_ids)} GPU(s) in train.device, but "
                f"only {visible_device_count} CUDA device(s) are visible. "
                "Please update train.device in configs/train.yaml."
            )
        print(f"CUDA_VISIBLE_DEVICES={device}")


def build_versioned_output_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    version_index = 1
    while True:
        output_dir = output_root / f"version{version_index}"
        if not output_dir.exists():
            return output_dir
        version_index += 1


def build_checkpoints_dir(output_dir: Path) -> Path:
    return output_dir / CHECKPOINTS_DIR_NAME


def resolve_checkpoint_path(checkpoint_text: str, output_root: Path) -> Path:
    checkpoint_path = Path(checkpoint_text).expanduser()
    candidates: list[Path] = []
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
        candidates.extend(
            [
                (PROJECT_ROOT / checkpoint_path).resolve(),
                (output_root / checkpoint_path).resolve(),
            ]
        )
        parts = checkpoint_path.parts
        if (
            len(parts) == 2
            and parts[0].startswith("version")
            and parts[1].startswith("checkpoint-")
        ):
            candidates.append(
                (output_root / parts[0] / CHECKPOINTS_DIR_NAME / parts[1]).resolve()
            )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def resolve_output_dir(train_args: TrainArgs) -> tuple[Path, Optional[Path], bool]:
    resume_checkpoint = normalize_optional_path(train_args.resume_from_checkpoint)
    if resume_checkpoint is None:
        output_root = resolve_path(train_args.output_dir)
        output_dir = build_versioned_output_dir(output_root)
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir, None, True

    output_root = resolve_path(train_args.output_dir)
    checkpoint_path = resolve_checkpoint_path(resume_checkpoint, output_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_dir():
        raise ValueError(f"Resume checkpoint must be a directory: {checkpoint_path}")
    if not checkpoint_path.name.startswith("checkpoint-"):
        raise ValueError(
            f"Resume path must point to a checkpoint-* directory: {checkpoint_path}"
        )
    if checkpoint_path.parent.name == CHECKPOINTS_DIR_NAME:
        output_dir = checkpoint_path.parent.parent
    else:
        output_dir = checkpoint_path.parent
    if not output_dir.name.startswith("version"):
        raise ValueError(f"Checkpoint parent must be a version directory: {output_dir}")
    return output_dir, checkpoint_path, False


def make_writable(path: Path) -> None:
    if path.exists():
        path.chmod(0o644)


def make_read_only(path: Path) -> None:
    path.chmod(0o444)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_writable(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def load_cumulative_train_runtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    return float(
        metrics.get("cumulative_train_runtime", metrics.get("train_runtime", 0.0))
    )


def copy_artifact_to_version_root(
    source_dir: Path,
    output_dir: Path,
    filename: str,
) -> None:
    source_path = source_dir / filename
    if not source_path.exists():
        return
    target_path = output_dir / filename
    make_writable(target_path)
    shutil.copy2(source_path, target_path)


def save_config_snapshot(config_path: Path, output_dir: Path) -> None:
    resolved_config_path = resolve_path(str(config_path))
    snapshot_path = output_dir / "train.yaml"
    make_writable(snapshot_path)
    shutil.copy2(resolved_config_path, snapshot_path)
    make_read_only(snapshot_path)


def save_tokenizer_snapshot(tokenizer, output_dir: Path) -> None:
    output_path = output_dir / "tokenizer_vocab.json"
    write_json(
        output_path,
        {
            "id_to_token": tokenizer.id_to_token,
            "token_to_id": tokenizer.token_to_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "mask_token_id": tokenizer.mask_token_id,
            "sep_token_id": tokenizer.sep_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": tokenizer.vocab_size,
        },
    )
    make_read_only(output_path)


def build_runtime_config_payload(
    *,
    model_args: LlamaTrainModelArgs,
    train_args: TrainArgs,
    schedule: ResolvedSchedule,
    llama_args: ModelArgs,
    train_csv: Path,
    output_dir: Path,
    resume_checkpoint: Optional[Path],
    is_new_run: bool,
) -> dict:
    return {
        "model_name": model_args.name,
        "config_snapshot": str(output_dir / "train.yaml"),
        "is_new_run": bool(is_new_run),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "train": asdict(train_args),
        "model": asdict(model_args),
        "resolved_train_csv": str(train_csv),
        "resolved_val_csv": (
            str(resolve_path(train_args.val_csv)) if train_args.val_csv else None
        ),
        "resolved_schedule": asdict(schedule),
        "llama_config": asdict(llama_args),
    }


def build_model_args(tokenizer, model_args: LlamaTrainModelArgs) -> ModelArgs:
    return ModelArgs(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_position_embeddings=model_args.max_position_embeddings,
        num_hidden_layers=model_args.num_hidden_layers,
        hidden_size=model_args.hidden_size,
        intermediate_size=model_args.intermediate_size,
        num_attention_heads=model_args.num_attention_heads,
        num_key_value_heads=model_args.num_key_value_heads,
        attention_dropout=model_args.attention_dropout,
        rms_norm_eps=model_args.rms_norm_eps,
        rope_theta=model_args.rope_theta,
        use_cache=False,
    )


def infer_device_count(train_args: TrainArgs) -> int:
    world_size_text = os.environ.get("WORLD_SIZE")
    if world_size_text:
        try:
            world_size = int(world_size_text)
        except ValueError:
            world_size = 0
        if world_size > 0:
            return world_size
    if train_args.device:
        device_ids = [
            item.strip()
            for item in str(train_args.device).split(",")
            if item.strip()
        ]
        if device_ids:
            return len(device_ids)
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def epoch_interval_to_steps(
    interval_epochs: float,
    steps_per_epoch: int,
    field_name: str,
) -> int:
    if interval_epochs <= 0:
        raise ValueError(f"{field_name} must be > 0, got {interval_epochs}")
    return max(1, int(math.ceil(interval_epochs * steps_per_epoch)))


def resolve_schedule(train_args: TrainArgs) -> ResolvedSchedule:
    train_csv = resolve_path(train_args.train_csv)
    train_samples = count_indexed_samples(train_csv, train_args.max_length)
    if train_samples <= 0:
        raise ValueError(f"No valid training samples found in {train_csv}")
    device_count = infer_device_count(train_args)
    global_batch_size = (
        int(train_args.batch_size)
        * int(train_args.gradient_accumulation_steps)
        * max(1, device_count)
    )
    steps_per_epoch = max(1, int(math.ceil(train_samples / global_batch_size)))

    save_strategy = str(train_args.save_strategy).lower()
    if save_strategy == "epoch":
        resolved_save_strategy = "steps"
        save_steps = epoch_interval_to_steps(
            train_args.save_every_epoch,
            steps_per_epoch,
            "save_every_epoch",
        )
    elif save_strategy == "no":
        resolved_save_strategy = "no"
        save_steps = None
    else:
        raise ValueError(
            f"save_strategy must be 'epoch' or 'no', got {train_args.save_strategy}"
        )

    eval_samples = 0
    eval_strategy = str(train_args.eval_strategy).lower()
    if train_args.val_csv and eval_strategy == "epoch":
        eval_csv = resolve_path(train_args.val_csv)
        eval_samples = count_indexed_samples(eval_csv, train_args.max_length)
        resolved_eval_strategy = "steps"
        eval_steps = epoch_interval_to_steps(
            train_args.eval_every_epoch,
            steps_per_epoch,
            "eval_every_epoch",
        )
    elif eval_strategy == "no" or not train_args.val_csv:
        resolved_eval_strategy = "no"
        eval_steps = None
    else:
        raise ValueError(
            f"eval_strategy must be 'epoch' or 'no', got {train_args.eval_strategy}"
        )
    return ResolvedSchedule(
        train_samples=train_samples,
        eval_samples=eval_samples,
        device_count=device_count,
        global_batch_size=global_batch_size,
        steps_per_epoch=steps_per_epoch,
        max_steps=train_args.max_steps,
        save_strategy=resolved_save_strategy,
        save_steps=save_steps,
        eval_strategy=resolved_eval_strategy,
        eval_steps=eval_steps,
    )


class InitialEvalAtFirstTrainLogCallback(TrainerCallback):
    def __init__(self) -> None:
        self.eval_step: Optional[int] = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_step is None and control.should_log:
            self.eval_step = int(state.global_step)
            control.should_evaluate = True
            if state.is_world_process_zero:
                print(
                    "Running initial validation alongside the first train loss "
                    f"at step {self.eval_step}"
                )
        return control

    def is_initial_eval_step(self, step: int) -> bool:
        return self.eval_step is not None and int(step) == self.eval_step


class BestEvalModelCallback(TrainerCallback):
    def __init__(
        self,
        output_dir: Path,
        early_stopping_patience: int = 3,
        initial_eval_callback: Optional[InitialEvalAtFirstTrainLogCallback] = None,
    ) -> None:
        self.best_model_dir = output_dir / "best_model"
        self.metrics_path = self.best_model_dir / "metrics.json"
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_counter = 0
        self.best_eval_loss = self._load_existing_best_eval_loss()
        self.initial_eval_callback = initial_eval_callback

    def _load_existing_best_eval_loss(self) -> Optional[float]:
        if not self.metrics_path.exists():
            return None
        with self.metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        value = metrics.get("best_eval_loss")
        return float(value) if value is not None else None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or model is None or not state.is_world_process_zero:
            return control
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control
        if (
            self.initial_eval_callback is not None
            and self.initial_eval_callback.is_initial_eval_step(state.global_step)
        ):
            print("Initial validation was logged")
            return control
        eval_loss = float(eval_loss)
        if self.best_eval_loss is not None and eval_loss >= self.best_eval_loss:
            self.early_stopping_counter += 1
            if (
                self.early_stopping_patience > 0
                and self.early_stopping_counter >= self.early_stopping_patience
            ):
                control.should_training_stop = True
            return control
        self.best_eval_loss = eval_loss
        self.early_stopping_counter = 0
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = unwrap_model(model, recursive=True)
        model_to_save.save_pretrained(
            str(self.best_model_dir),
            safe_serialization=bool(args.save_safetensors),
        )
        torch.save(args, self.best_model_dir / "training_args.bin")
        write_json(
            self.metrics_path,
            {
                "best_eval_loss": eval_loss,
                "best_step": int(state.global_step),
                "best_epoch": float(state.epoch) if state.epoch is not None else None,
                "metric_name": "eval_loss",
                "selection": "lowest",
                "source": "eval",
                "saved_from": "trainer_callback",
                "early_stopping_patience": self.early_stopping_patience,
            },
        )
        return control


def build_training_arguments(
    train_args: TrainArgs,
    schedule: ResolvedSchedule,
    output_dir: Path,
) -> TrainingArguments:
    kwargs = {
        "output_dir": str(build_checkpoints_dir(output_dir)),
        "overwrite_output_dir": True,
        "max_steps": int(schedule.max_steps),
        "learning_rate": float(train_args.learning_rate),
        "weight_decay": float(train_args.weight_decay),
        "per_device_train_batch_size": int(train_args.batch_size),
        "per_device_eval_batch_size": int(train_args.batch_size),
        "gradient_accumulation_steps": int(train_args.gradient_accumulation_steps),
        "logging_strategy": "steps",
        "logging_steps": int(train_args.logging_steps),
        "save_strategy": schedule.save_strategy,
        "save_total_limit": int(train_args.save_total_limit),
        "warmup_ratio": float(train_args.warmup_ratio),
        "remove_unused_columns": False,
        "dataloader_num_workers": int(train_args.num_workers),
        "seed": int(train_args.seed),
        "data_seed": int(train_args.seed),
        "report_to": "none",
        "save_safetensors": True,
        "fp16": bool(train_args.fp16) and torch.cuda.is_available(),
        "bf16": bool(train_args.bf16) and torch.cuda.is_available(),
        "logging_dir": str(output_dir / "logs"),
    }
    if schedule.save_strategy == "steps" and schedule.save_steps is not None:
        kwargs["save_steps"] = int(schedule.save_steps)
    if train_args.val_csv and schedule.eval_strategy != "no":
        kwargs["eval_strategy"] = schedule.eval_strategy
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
        if schedule.eval_strategy == "steps" and schedule.eval_steps is not None:
            kwargs["eval_steps"] = int(schedule.eval_steps)
    else:
        kwargs["eval_strategy"] = "no"
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" not in signature.parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return TrainingArguments(**kwargs)


def has_eval_at_step(log_history: Sequence[dict], step: int) -> bool:
    for item in reversed(log_history):
        if "eval_loss" in item:
            return int(item.get("step", -1)) == int(step)
    return False


def build_data_args(
    csv_path: str,
    train_args: TrainArgs,
    *,
    is_training: bool,
) -> DataArgs:
    return DataArgs(
        csv_path=str(resolve_path(csv_path)),
        max_length=train_args.max_length,
        batch_size=train_args.batch_size,
        num_workers=train_args.num_workers,
        drop_last=False,
        mixed_infill_enabled=(
            is_training and train_args.mixed_infill_enabled
        ),
        vhh_span_probability=train_args.vhh_span_probability,
        min_span_length=train_args.min_span_length,
        max_span_length=train_args.max_span_length,
        task_seed=train_args.task_seed,
        span_length_seed=train_args.span_length_seed,
        span_position_seed=train_args.span_position_seed,
        cdr3_sampling_enabled=(
            is_training and train_args.cdr3_sampling_enabled
        ),
        cdr3_full_keep_threshold=train_args.cdr3_full_keep_threshold,
        cdr3_sampling_power=train_args.cdr3_sampling_power,
        cdr3_sampling_seed=train_args.cdr3_sampling_seed,
    )


class FullEpochShuffleTrainer(Trainer):
    def __init__(self, *args, train_sampler=None, **kwargs):
        self.train_sampler = train_sampler
        self.component_loss_sums = {"cdr3": 0.0, "vhh": 0.0}
        self.component_loss_steps = {"cdr3": 0, "vhh": 0}
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None
        if self.train_sampler is not None:
            return self.train_sampler
        data_seed = self.args.data_seed
        seed = self.args.seed if data_seed is None else data_seed
        return EpochRandomSampler(train_dataset, seed=int(seed))

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        task_ids = inputs.pop("task_id")
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = outputs.loss

        if model.training:
            with torch.no_grad():
                shift_logits = outputs.logits[..., :-1, :].detach().float()
                shift_labels = labels[..., 1:].contiguous()
                token_losses = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).reshape_as(shift_labels)
                valid_tokens = shift_labels.ne(-100)
                cdr3_tokens = valid_tokens & task_ids[:, None].eq(TASK_CDR3)
                vhh_tokens = valid_tokens & task_ids[:, None].eq(TASK_VHH)
                component_stats = torch.stack(
                    (
                        token_losses[cdr3_tokens].sum(),
                        cdr3_tokens.sum().to(token_losses.dtype),
                        token_losses[vhh_tokens].sum(),
                        vhh_tokens.sum().to(token_losses.dtype),
                    )
                )
                component_stats = self.accelerator.reduce(
                    component_stats,
                    reduction="sum",
                )
                for name, offset in (("cdr3", 0), ("vhh", 2)):
                    token_count = component_stats[offset + 1].item()
                    if token_count > 0:
                        component_loss = component_stats[offset].item() / token_count
                        self.component_loss_sums[name] += component_loss
                        self.component_loss_steps[name] += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        if "loss" in logs:
            logs = dict(logs)
            for name in ("cdr3", "vhh"):
                step_count = self.component_loss_steps[name]
                if step_count > 0:
                    logs[f"{name}_loss"] = round(
                        self.component_loss_sums[name] / step_count,
                        4,
                    )
                self.component_loss_sums[name] = 0.0
                self.component_loss_steps[name] = 0
        super().log(logs, *args, **kwargs)


def train(
    train_args: TrainArgs,
    model_args: LlamaTrainModelArgs,
    config_path: Path,
) -> Path:
    configure_cuda_devices(train_args.device)
    set_seed(train_args.seed)

    train_csv = resolve_path(train_args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Train csv not found: {train_csv}")
    if train_args.val_csv and not resolve_path(train_args.val_csv).exists():
        raise FileNotFoundError(
            f"Validation csv not found: {resolve_path(train_args.val_csv)}"
        )

    schedule = resolve_schedule(train_args)
    output_dir, resume_checkpoint, is_new_run = resolve_output_dir(train_args)
    build_checkpoints_dir(output_dir).mkdir(parents=True, exist_ok=True)
    all_results_path = output_dir / "all_results.json"
    previous_train_runtime = load_cumulative_train_runtime(all_results_path)

    tokenizer = build_default_tokenizer()
    llama_args = build_model_args(tokenizer=tokenizer, model_args=model_args)
    model = build_model(llama_args)
    model.config.name_or_path = model_args.name
    model.config.use_cache = False

    train_data_args = build_data_args(
        str(train_csv),
        train_args,
        is_training=True,
    )
    train_dataset = VHHIndexedDataset(
        args=train_data_args,
        tokenizer=tokenizer,
        expected_length=schedule.train_samples,
    )
    train_sampler = None
    if train_args.cdr3_sampling_enabled:
        train_sampler = CDR3SubsampleSampler(
            train_dataset,
            full_keep_threshold=train_args.cdr3_full_keep_threshold,
            sampling_power=train_args.cdr3_sampling_power,
            seed=train_args.cdr3_sampling_seed,
        )

    sampled_epoch_samples = (
        len(train_sampler) if train_sampler is not None else schedule.train_samples
    )
    schedule.steps_per_epoch = max(
        1,
        int(math.ceil(sampled_epoch_samples / schedule.global_batch_size)),
    )
    schedule.max_steps = train_args.max_steps
    if schedule.save_strategy == "steps":
        schedule.save_steps = epoch_interval_to_steps(
            train_args.save_every_epoch,
            schedule.steps_per_epoch,
            "save_every_epoch",
        )
    if schedule.eval_strategy == "steps":
        schedule.eval_steps = epoch_interval_to_steps(
            train_args.eval_every_epoch,
            schedule.steps_per_epoch,
            "eval_every_epoch",
        )

    eval_dataset = None
    if train_args.val_csv and schedule.eval_strategy != "no":
        val_data_args = build_data_args(
            train_args.val_csv,
            train_args,
            is_training=False,
        )
        eval_dataset = VHHIndexedDataset(
            args=val_data_args,
            tokenizer=tokenizer,
            expected_length=schedule.eval_samples,
        )

    data_collator = lambda batch: collate_fn(
        batch,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=train_data_args.ignore_index,
    )

    save_config_snapshot(config_path=config_path, output_dir=output_dir)
    runtime_config_path = output_dir / "runtime_config.json"
    runtime_config = build_runtime_config_payload(
        model_args=model_args,
        train_args=train_args,
        schedule=schedule,
        llama_args=llama_args,
        train_csv=train_csv,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        is_new_run=is_new_run,
    )
    runtime_config["cdr3_sampling"] = {
        "enabled": train_args.cdr3_sampling_enabled,
        "full_dataset_samples": schedule.train_samples,
        "sampled_epoch_samples": sampled_epoch_samples,
        "sampled_steps_per_epoch": schedule.steps_per_epoch,
        "expected_sampler_epochs": schedule.max_steps / schedule.steps_per_epoch,
        "schedule_basis": "explicit_max_steps",
    }
    write_json(runtime_config_path, runtime_config)
    make_read_only(runtime_config_path)
    if is_new_run:
        save_tokenizer_snapshot(tokenizer, output_dir)

    print(f"Output directory: {output_dir}")
    print(f"Model name: {model_args.name}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Train samples: {schedule.train_samples}")
    print(f"Train max steps: {train_args.max_steps}")
    print(f"Visible devices: {schedule.device_count}")
    print(f"Global batch size: {schedule.global_batch_size}")
    print(f"Steps per epoch: {schedule.steps_per_epoch}")
    print(f"Resolved max_steps: {schedule.max_steps}")
    print(f"Mixed infill enabled: {train_args.mixed_infill_enabled}")
    if train_args.mixed_infill_enabled:
        print(f"VHH span probability: {train_args.vhh_span_probability}")
        print(
            "VHH span length: "
            f"{train_args.min_span_length}-{train_args.max_span_length}"
        )
    print(f"CDR3 sampling enabled: {train_args.cdr3_sampling_enabled}")
    if train_sampler is not None:
        print(
            "CDR3 sampling: "
            f"threshold={train_args.cdr3_full_keep_threshold}, "
            f"sampling_power={train_args.cdr3_sampling_power}, "
            f"sampled epoch={len(train_sampler):,} sequences "
            f"({len(train_sampler) / schedule.train_samples:.2%})"
        )
        print(
            "Expected sampler epochs: "
            f"{schedule.max_steps / schedule.steps_per_epoch:.2f}"
        )
    if schedule.save_strategy == "steps" and schedule.save_steps is not None:
        print(f"Resolved save_steps: {schedule.save_steps}")
    if schedule.eval_strategy == "steps" and schedule.eval_steps is not None:
        print(f"Resolved eval_steps: {schedule.eval_steps}")

    training_args = build_training_arguments(
        train_args=train_args,
        schedule=schedule,
        output_dir=output_dir,
    )
    callbacks = None
    if eval_dataset is not None:
        initial_eval_callback = (
            InitialEvalAtFirstTrainLogCallback() if is_new_run else None
        )
        callbacks = []
        if initial_eval_callback is not None:
            callbacks.append(initial_eval_callback)
        callbacks.append(
            BestEvalModelCallback(
                output_dir,
                early_stopping_patience=train_args.early_stopping_patience,
                initial_eval_callback=initial_eval_callback,
            )
        )

    trainer = FullEpochShuffleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        train_sampler=train_sampler,
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint else None
        ),
    )

    final_step = int(trainer.state.global_step)
    if eval_dataset is not None and not has_eval_at_step(
        trainer.state.log_history,
        final_step,
    ):
        print(f"Running final evaluation at step {final_step}")
        final_eval_metrics = trainer.evaluate()
        print(f"Final eval loss: {final_eval_metrics['eval_loss']}")

    trainer.save_model(str(output_dir / "final_model"))
    trainer.save_state()
    training_output_dir = Path(training_args.output_dir)
    copy_artifact_to_version_root(
        training_output_dir,
        output_dir,
        "trainer_state.json",
    )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    copy_artifact_to_version_root(
        training_output_dir,
        output_dir,
        "train_results.json",
    )
    copy_artifact_to_version_root(
        training_output_dir,
        output_dir,
        "all_results.json",
    )
    with all_results_path.open(encoding="utf-8") as handle:
        all_results = json.load(handle)
    cumulative_train_runtime = previous_train_runtime + float(
        train_result.metrics["train_runtime"]
    )
    all_results["cumulative_train_runtime"] = cumulative_train_runtime
    all_results["cumulative_train_runtime_hours"] = cumulative_train_runtime / 3600.0
    write_json(all_results_path, all_results)
    return output_dir


def parse_args() -> tuple[TrainArgs, LlamaTrainModelArgs, Path]:
    parser = argparse.ArgumentParser(
        description="Train VHH-Llama with mixed CDR3/VHH span infilling."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    args = parser.parse_args()
    config_path = resolve_path(str(args.config))
    train_args, model_args = build_args_from_config(
        load_yaml_config(config_path)
    )
    return train_args, model_args, config_path


def main() -> None:
    train_args, model_args, config_path = parse_args()
    train(train_args=train_args, model_args=model_args, config_path=config_path)


if __name__ == "__main__":
    main()
