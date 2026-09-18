from __future__ import annotations

import json
import math
import mmap
import os
import random
import struct
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import Dataset, Sampler


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[SEP]", "[EOS]"]
CSV_COLUMNS = ("CDR3_seq", "VHH_seq", "L_seq", "R_seq")
OFFSET_INDEX_VERSION = 1
OFFSET_STRUCT = struct.Struct("<Q")
INDEX_WRITE_CHUNK_SIZE = 1_000_000


class SimpleAATokenizer:
    def __init__(self) -> None:
        self.id_to_token = SPECIAL_TOKENS + AMINO_ACIDS
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.id_to_token)
        }
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        self.sep_token = "[SEP]"
        self.eos_token = "[EOS]"
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.mask_token_id = self.token_to_id[self.mask_token]
        self.sep_token_id = self.token_to_id[self.sep_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.vocab_size = len(self.id_to_token)

    def encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

    def decode_ids(self, token_ids: Sequence[int]) -> List[str]:
        return [self.id_to_token[token_id] for token_id in token_ids]


@dataclass
class BaseDataArgs:
    csv_path: str
    max_length: int = 256
    batch_size: int = 32
    num_workers: int = 0
    ignore_index: int = -100
    drop_last: bool = False


def valid_sequences(
    l_seq,
    r_seq,
    cdr3_seq,
    vhh_seq,
    max_length: int,
) -> bool:
    return (
        bool(l_seq and r_seq and cdr3_seq and vhh_seq)
        and l_seq + cdr3_seq + r_seq == vhh_seq
        and len(l_seq) + len(r_seq) + len(cdr3_seq) + 3 <= max_length
    )


def offset_index_paths(
    csv_path: Path,
    max_length: int,
    index_dir: Path | None = None,
) -> tuple[Path, Path]:
    index_root = csv_path.parent if index_dir is None else Path(index_dir)
    prefix = f"{csv_path.name}.max_length_{max_length}"
    return (
        index_root / f"{prefix}.offsets.bin",
        index_root / f"{prefix}.offsets.json",
    )


def load_offset_index_count(
    csv_path: Path,
    max_length: int,
    index_path: Path,
    metadata_path: Path,
) -> int | None:
    if not index_path.exists() or not metadata_path.exists():
        return None
    try:
        csv_stat = csv_path.stat()
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        sample_count = int(metadata["sample_count"])
        valid = (
            metadata.get("version") == OFFSET_INDEX_VERSION
            and metadata.get("csv_size") == csv_stat.st_size
            and metadata.get("csv_mtime_ns") == csv_stat.st_mtime_ns
            and metadata.get("max_length") == max_length
            and metadata.get("offset_format") == "uint64_le"
            and sample_count > 0
            and index_path.stat().st_size == sample_count * OFFSET_STRUCT.size
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return sample_count if valid else None


def write_offset_chunk(handle, offsets: array) -> None:
    if not offsets:
        return
    if OFFSET_STRUCT.format == "<Q" and sys.byteorder != "little":
        offsets.byteswap()
    offsets.tofile(handle)


def build_offset_index(
    csv_path: Path,
    max_length: int,
    index_path: Path,
    metadata_path: Path,
) -> int:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    csv_stat = csv_path.stat()
    temporary_index = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.{os.getpid()}.tmp"
    )
    scanned = 0
    sample_count = 0
    offsets = array("Q")
    started_at = time.monotonic()
    print(f"Building row offset index for {csv_path} ...", flush=True)
    try:
        with (
            csv_path.open("rb", buffering=1024 * 1024) as source,
            temporary_index.open("wb", buffering=1024 * 1024) as output,
        ):
            header = source.readline().rstrip(b"\r\n").decode("ascii").split(",")
            if tuple(header) != CSV_COLUMNS:
                raise ValueError(f"Unexpected CSV header in {csv_path}: {header}")
            while True:
                offset = source.tell()
                raw_line = source.readline()
                if not raw_line:
                    break
                scanned += 1
                fields = raw_line.rstrip(b"\r\n").split(b",")
                if len(fields) != len(CSV_COLUMNS):
                    raise ValueError(
                        f"{csv_path}:{scanned + 1}: expected four CSV columns"
                    )
                cdr3_seq, vhh_seq, l_seq, r_seq = fields
                if valid_sequences(
                    l_seq=l_seq,
                    r_seq=r_seq,
                    cdr3_seq=cdr3_seq,
                    vhh_seq=vhh_seq,
                    max_length=max_length,
                ):
                    offsets.append(offset)
                    sample_count += 1
                if len(offsets) == INDEX_WRITE_CHUNK_SIZE:
                    write_offset_chunk(output, offsets)
                    offsets = array("Q")
                if scanned % 1_000_000 == 0:
                    elapsed = time.monotonic() - started_at
                    print(
                        f"Offset index progress: scanned={scanned:,}, "
                        f"valid={sample_count:,}, elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            write_offset_chunk(output, offsets)

        final_csv_stat = csv_path.stat()
        if (
            final_csv_stat.st_size != csv_stat.st_size
            or final_csv_stat.st_mtime_ns != csv_stat.st_mtime_ns
        ):
            raise RuntimeError(f"CSV changed while its offset index was built: {csv_path}")
        if sample_count <= 0:
            raise ValueError(f"No valid samples found in {csv_path}")
        os.replace(temporary_index, index_path)
        with temporary_metadata.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": OFFSET_INDEX_VERSION,
                    "csv_size": csv_stat.st_size,
                    "csv_mtime_ns": csv_stat.st_mtime_ns,
                    "max_length": max_length,
                    "offset_format": "uint64_le",
                    "sample_count": sample_count,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary_index.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    print(
        f"Offset index complete: scanned={scanned:,}, valid={sample_count:,}, "
        f"elapsed={time.monotonic() - started_at:.1f}s, path={index_path}",
        flush=True,
    )
    return sample_count


def ensure_offset_index(
    csv_path: str | Path,
    max_length: int,
    index_dir: str | Path | None = None,
) -> tuple[Path, Path, int]:
    resolved_csv_path = Path(csv_path).resolve()
    resolved_index_dir = None if index_dir is None else Path(index_dir).resolve()
    index_path, metadata_path = offset_index_paths(
        resolved_csv_path,
        max_length,
        resolved_index_dir,
    )
    sample_count = load_offset_index_count(
        resolved_csv_path,
        max_length,
        index_path,
        metadata_path,
    )
    if sample_count is None:
        sample_count = build_offset_index(
            resolved_csv_path,
            max_length,
            index_path,
            metadata_path,
        )
    else:
        print(
            f"Using row offset index for {resolved_csv_path}: "
            f"{sample_count:,} samples",
            flush=True,
        )
    return index_path, metadata_path, sample_count


def count_indexed_samples(csv_path: str | Path, max_length: int) -> int:
    return ensure_offset_index(csv_path, max_length)[2]


class BaseVHHIndexedDataset(Dataset):
    def __init__(
        self,
        args: BaseDataArgs,
        tokenizer: SimpleAATokenizer,
        expected_length: int | None = None,
        index_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.csv_path = Path(args.csv_path).resolve()
        self.index_path, self.metadata_path, self.length = ensure_offset_index(
            self.csv_path,
            args.max_length,
            index_dir=index_dir,
        )
        if expected_length is not None and self.length != expected_length:
            raise ValueError(
                f"Indexed sample count mismatch for {self.csv_path}: "
                f"expected {expected_length}, found {self.length}"
            )
        self._csv_file = None
        self._index_file = None
        self._csv_mmap = None
        self._index_mmap = None

    def __len__(self) -> int:
        return self.length

    def _ensure_open(self) -> None:
        if self._csv_mmap is not None:
            return
        self._csv_file = self.csv_path.open("rb")
        self._index_file = self.index_path.open("rb")
        self._csv_mmap = mmap.mmap(
            self._csv_file.fileno(), length=0, access=mmap.ACCESS_READ
        )
        self._index_mmap = mmap.mmap(
            self._index_file.fileno(), length=0, access=mmap.ACCESS_READ
        )

    def close(self) -> None:
        for attribute in ("_csv_mmap", "_index_mmap"):
            resource = getattr(self, attribute, None)
            if resource is not None:
                resource.close()
                setattr(self, attribute, None)
        for attribute in ("_csv_file", "_index_file"):
            resource = getattr(self, attribute, None)
            if resource is not None:
                resource.close()
                setattr(self, attribute, None)

    def __getstate__(self):
        state = self.__dict__.copy()
        for attribute in ("_csv_file", "_index_file", "_csv_mmap", "_index_mmap"):
            state[attribute] = None
        return state

    def __del__(self) -> None:
        self.close()


class BaseEpochRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sequence, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(self.epoch, int(epoch))

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(
            len(self.data_source), generator=generator, dtype=torch.int32
        )
        for start in range(0, len(permutation), SAMPLER_CONVERSION_CHUNK_SIZE):
            yield from permutation[
                start : start + SAMPLER_CONVERSION_CHUNK_SIZE
            ].tolist()


def base_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    max_length = max(item["input_ids"].size(0) for item in batch)
    input_id_rows = []
    attention_mask_rows = []
    label_rows = []
    for item in batch:
        pad_length = max_length - item["input_ids"].size(0)
        input_id_rows.append(
            torch.nn.functional.pad(
                item["input_ids"], (0, pad_length), value=pad_token_id
            )
        )
        attention_mask_rows.append(
            torch.nn.functional.pad(
                item["attention_mask"], (0, pad_length), value=0
            )
        )
        label_rows.append(
            torch.nn.functional.pad(
                item["labels"], (0, pad_length), value=ignore_index
            )
        )
    return {
        "input_ids": torch.stack(input_id_rows),
        "attention_mask": torch.stack(attention_mask_rows),
        "labels": torch.stack(label_rows),
    }


def build_default_tokenizer() -> SimpleAATokenizer:
    return SimpleAATokenizer()


CDR3_GROUP_INDEX_VERSION = 1
UINT32_STRUCT = struct.Struct("<I")
UINT64_STRUCT = struct.Struct("<Q")
SAMPLER_CONVERSION_CHUNK_SIZE = 65_536
TASK_CDR3 = 0
TASK_VHH = 1


@dataclass
class DataArgs(BaseDataArgs):
    mixed_infill_enabled: bool = False
    vhh_span_probability: float = 0.2
    min_span_length: int = 5
    max_span_length: int = 15
    task_seed: int = 1
    span_length_seed: int = 2
    span_position_seed: int = 3
    cdr3_sampling_enabled: bool = False
    cdr3_full_keep_threshold: int = 100
    cdr3_sampling_power: float = 0.3
    cdr3_sampling_seed: int = 10_000_061


def cdr3_group_index_paths(
    csv_path: Path,
    max_length: int,
) -> tuple[Path, Path, Path]:
    prefix = f"{csv_path.name}.max_length_{max_length}.cdr3_groups"
    return (
        csv_path.parent / f"{prefix}.offsets.bin",
        csv_path.parent / f"{prefix}.members.bin",
        csv_path.parent / f"{prefix}.json",
    )


def load_cdr3_group_index_metadata(
    dataset: BaseVHHIndexedDataset,
    offsets_path: Path,
    members_path: Path,
    metadata_path: Path,
) -> dict | None:
    if not offsets_path.exists() or not members_path.exists() or not metadata_path.exists():
        return None

    try:
        csv_stat = dataset.csv_path.stat()
        offset_index_stat = dataset.index_path.stat()
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        group_count = int(metadata["group_count"])
        sample_count = int(metadata["sample_count"])
        valid = (
            metadata.get("version") == CDR3_GROUP_INDEX_VERSION
            and metadata.get("csv_size") == csv_stat.st_size
            and metadata.get("csv_mtime_ns") == csv_stat.st_mtime_ns
            and metadata.get("offset_index_size") == offset_index_stat.st_size
            and metadata.get("offset_index_mtime_ns") == offset_index_stat.st_mtime_ns
            and metadata.get("max_length") == dataset.args.max_length
            and sample_count == len(dataset)
            and group_count > 0
            and offsets_path.stat().st_size == (group_count + 1) * UINT64_STRUCT.size
            and members_path.stat().st_size == sample_count * UINT32_STRUCT.size
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return metadata if valid else None


def _write_native_array(handle, values: array) -> None:
    if sys.byteorder != "little":
        values.byteswap()
    values.tofile(handle)


def build_cdr3_group_index(
    dataset: BaseVHHIndexedDataset,
    offsets_path: Path,
    members_path: Path,
    metadata_path: Path,
) -> dict:
    if len(dataset) >= 2**32:
        raise ValueError("CDR3 group index supports fewer than 2^32 samples")

    offsets_tmp = offsets_path.with_name(f".{offsets_path.name}.{os.getpid()}.tmp")
    members_tmp = members_path.with_name(f".{members_path.name}.{os.getpid()}.tmp")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    group_ids: dict[bytes, int] = {}
    group_counts = array("I")
    row_group_ids = array("I")
    started_at = time.monotonic()

    print(f"Building CDR3 group index for {dataset.csv_path} ...", flush=True)
    try:
        with (
            dataset.csv_path.open("rb") as csv_file,
            dataset.index_path.open("rb") as index_file,
        ):
            csv_mmap = mmap.mmap(csv_file.fileno(), length=0, access=mmap.ACCESS_READ)
            index_mmap = mmap.mmap(index_file.fileno(), length=0, access=mmap.ACCESS_READ)
            try:
                for sample_index in range(len(dataset)):
                    row_offset = OFFSET_STRUCT.unpack_from(
                        index_mmap,
                        sample_index * OFFSET_STRUCT.size,
                    )[0]
                    csv_mmap.seek(row_offset)
                    cdr3_seq = csv_mmap.readline().split(b",", 1)[0]
                    group_id = group_ids.get(cdr3_seq)
                    if group_id is None:
                        group_id = len(group_counts)
                        group_ids[cdr3_seq] = group_id
                        group_counts.append(0)
                    group_counts[group_id] += 1
                    row_group_ids.append(group_id)

                    if (sample_index + 1) % 1_000_000 == 0:
                        print(
                            "CDR3 group index progress: "
                            f"rows={sample_index + 1:,}, groups={len(group_counts):,}",
                            flush=True,
                        )
            finally:
                index_mmap.close()
                csv_mmap.close()

        group_offsets = array("Q", [0])
        cumulative = 0
        for count in group_counts:
            cumulative += count
            group_offsets.append(cumulative)

        with offsets_tmp.open("wb") as handle:
            _write_native_array(handle, group_offsets)

        with members_tmp.open("w+b") as handle:
            handle.truncate(len(dataset) * UINT32_STRUCT.size)
            members_mmap = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_WRITE)
            try:
                cursors = array("Q", group_offsets[:-1])
                for sample_index, group_id in enumerate(row_group_ids):
                    member_position = cursors[group_id]
                    UINT32_STRUCT.pack_into(
                        members_mmap,
                        member_position * UINT32_STRUCT.size,
                        sample_index,
                    )
                    cursors[group_id] += 1
                members_mmap.flush()
            finally:
                members_mmap.close()

        csv_stat = dataset.csv_path.stat()
        offset_index_stat = dataset.index_path.stat()
        metadata = {
            "version": CDR3_GROUP_INDEX_VERSION,
            "csv_size": csv_stat.st_size,
            "csv_mtime_ns": csv_stat.st_mtime_ns,
            "offset_index_size": offset_index_stat.st_size,
            "offset_index_mtime_ns": offset_index_stat.st_mtime_ns,
            "max_length": dataset.args.max_length,
            "sample_count": len(dataset),
            "group_count": len(group_counts),
            "offset_format": "uint64_le",
            "member_format": "uint32_le",
        }
        with metadata_tmp.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

        os.replace(offsets_tmp, offsets_path)
        os.replace(members_tmp, members_path)
        os.replace(metadata_tmp, metadata_path)
    finally:
        offsets_tmp.unlink(missing_ok=True)
        members_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)

    print(
        f"CDR3 group index complete: samples={len(dataset):,}, "
        f"groups={metadata['group_count']:,}, "
        f"elapsed={time.monotonic() - started_at:.1f}s",
        flush=True,
    )
    return metadata


class CDR3GroupIndex:
    def __init__(self, dataset: BaseVHHIndexedDataset) -> None:
        self.offsets_path, self.members_path, self.metadata_path = (
            cdr3_group_index_paths(dataset.csv_path, dataset.args.max_length)
        )
        metadata = load_cdr3_group_index_metadata(
            dataset,
            self.offsets_path,
            self.members_path,
            self.metadata_path,
        )
        if metadata is None:
            metadata = build_cdr3_group_index(
                dataset,
                self.offsets_path,
                self.members_path,
                self.metadata_path,
            )
        else:
            print(
                f"Using CDR3 group index for {dataset.csv_path}: "
                f"{metadata['group_count']:,} groups",
                flush=True,
            )

        self.sample_count = int(metadata["sample_count"])
        self.group_count = int(metadata["group_count"])
        self._offsets_file = self.offsets_path.open("rb")
        self._members_file = self.members_path.open("rb")
        self._offsets_mmap = mmap.mmap(
            self._offsets_file.fileno(), length=0, access=mmap.ACCESS_READ
        )
        self._members_mmap = mmap.mmap(
            self._members_file.fileno(), length=0, access=mmap.ACCESS_COPY
        )

    def group_bounds(self, group_id: int) -> tuple[int, int]:
        start = UINT64_STRUCT.unpack_from(
            self._offsets_mmap, group_id * UINT64_STRUCT.size
        )[0]
        end = UINT64_STRUCT.unpack_from(
            self._offsets_mmap, (group_id + 1) * UINT64_STRUCT.size
        )[0]
        return start, end

    def members_tensor(self) -> torch.Tensor:
        return torch.frombuffer(self._members_mmap, dtype=torch.int32)


def cdr3_group_sample_count(
    group_size: int,
    full_keep_threshold: int,
    sampling_power: float,
) -> int:
    if group_size <= full_keep_threshold:
        return group_size
    selection_ratio = (full_keep_threshold / group_size) ** sampling_power
    return int(math.ceil(group_size * selection_ratio))


class CDR3SubsampleSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Sequence,
        *,
        full_keep_threshold: int,
        sampling_power: float,
        seed: int,
    ) -> None:
        self.data_source = data_source
        self.full_keep_threshold = int(full_keep_threshold)
        self.sampling_power = float(sampling_power)
        self.seed = int(seed)
        self.epoch = 0
        self.group_index = CDR3GroupIndex(data_source)
        self.sample_count = self.group_index.sample_count
        self.head_groups: list[tuple[int, int, int]] = []
        for group_id in range(self.group_index.group_count):
            start, end = self.group_index.group_bounds(group_id)
            group_size = end - start
            if group_size <= self.full_keep_threshold:
                continue
            keep_count = cdr3_group_sample_count(
                group_size,
                self.full_keep_threshold,
                self.sampling_power,
            )
            self.sample_count -= group_size - keep_count
            self.head_groups.append((start, end, keep_count))

    def __len__(self) -> int:
        return self.sample_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(self.epoch, int(epoch))
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(self.epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        members = self.group_index.members_tensor()
        selected = torch.empty(self.sample_count, dtype=torch.int32)
        selected_position = 0
        member_position = 0

        for start, end, keep_count in self.head_groups:
            preceding_count = start - member_position
            selected[
                selected_position : selected_position + preceding_count
            ] = members[member_position:start]
            selected_position += preceding_count
            group_size = end - start
            local_order = torch.randperm(
                group_size,
                generator=generator,
                dtype=torch.int64,
            )[:keep_count]
            chosen = members[start + local_order]
            selected[selected_position : selected_position + keep_count] = chosen
            selected_position += keep_count
            member_position = end

        remaining_count = self.group_index.sample_count - member_position
        selected[selected_position:] = members[member_position:]
        selected_position += remaining_count

        global_order = torch.randperm(
            self.sample_count,
            generator=generator,
            dtype=torch.int64,
        )
        for start in range(0, self.sample_count, SAMPLER_CONVERSION_CHUNK_SIZE):
            positions = global_order[start : start + SAMPLER_CONVERSION_CHUNK_SIZE]
            yield from selected[positions].tolist()


def build_infill_tokens(
    prefix: str,
    target: str,
    suffix: str,
    tokenizer: SimpleAATokenizer,
) -> tuple[List[str], int]:
    prompt_tokens = (
        list(prefix)
        + [tokenizer.mask_token]
        + list(suffix)
        + [tokenizer.sep_token]
    )
    answer_tokens = list(target) + [tokenizer.eos_token]
    return prompt_tokens + answer_tokens, len(prompt_tokens)


def select_infill_parts(
    row: Dict[str, str],
    args: DataArgs,
    sample_index: int,
    epoch: int,
) -> tuple[str, str, str, int]:
    if not args.mixed_infill_enabled:
        return row["L_seq"], row["CDR3_seq"], row["R_seq"], TASK_CDR3

    seed_offset = epoch * 1_000_003 + sample_index
    task_rng = random.Random(args.task_seed + seed_offset)
    if task_rng.random() >= args.vhh_span_probability:
        return row["L_seq"], row["CDR3_seq"], row["R_seq"], TASK_CDR3

    vhh_seq = row["VHH_seq"]
    max_span_length = min(args.max_span_length, len(vhh_seq))
    min_span_length = min(args.min_span_length, max_span_length)
    length_rng = random.Random(args.span_length_seed + seed_offset)
    span_length = length_rng.randint(min_span_length, max_span_length)
    position_rng = random.Random(args.span_position_seed + seed_offset)
    span_start = position_rng.randint(0, len(vhh_seq) - span_length)
    span_end = span_start + span_length
    return (
        vhh_seq[:span_start],
        vhh_seq[span_start:span_end],
        vhh_seq[span_end:],
        TASK_VHH,
    )


def encode_row(
    row: Dict[str, str],
    tokenizer: SimpleAATokenizer,
    args: DataArgs,
    sample_index: int = 0,
    epoch: int = 0,
) -> Dict[str, torch.Tensor] | None:
    l_seq = row["L_seq"]
    r_seq = row["R_seq"]
    cdr3_seq = row["CDR3_seq"]
    vhh_seq = row["VHH_seq"]

    if not l_seq or not r_seq or not cdr3_seq or not vhh_seq:
        return None
    if l_seq + cdr3_seq + r_seq != vhh_seq:
        return None

    prefix, target, suffix, task_id = select_infill_parts(
        row,
        args,
        sample_index=sample_index,
        epoch=epoch,
    )
    tokens, prompt_length = build_infill_tokens(
        prefix,
        target,
        suffix,
        tokenizer,
    )
    input_ids = tokenizer.encode_tokens(tokens)
    if len(input_ids) > args.max_length:
        return None

    labels = input_ids.copy()
    labels[:prompt_length] = [args.ignore_index] * prompt_length
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "task_id": torch.tensor(task_id, dtype=torch.long),
    }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    collated = base_collate_fn(
        batch,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
    )
    collated["task_id"] = torch.stack([item["task_id"] for item in batch])
    return collated


class VHHIndexedDataset(BaseVHHIndexedDataset):
    args: DataArgs

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(self.epoch, int(epoch))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)

        self._ensure_open()
        offset = OFFSET_STRUCT.unpack_from(
            self._index_mmap,
            index * OFFSET_STRUCT.size,
        )[0]
        self._csv_mmap.seek(offset)
        fields = self._csv_mmap.readline().rstrip(b"\r\n").split(b",")
        if len(fields) != len(CSV_COLUMNS):
            raise ValueError(f"Failed to parse indexed CSV row at byte offset {offset}")
        row = {
            name: value.decode("ascii")
            for name, value in zip(CSV_COLUMNS, fields)
        }
        sample = encode_row(
            row,
            self.tokenizer,
            self.args,
            sample_index=index,
            epoch=self.epoch,
        )
        if sample is None:
            raise ValueError(f"Indexed CSV row became invalid at byte offset {offset}")
        return sample


class EpochRandomSampler(BaseEpochRandomSampler):
    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(self.epoch)
