# VHH-Llama

VHH-Llama is a language model for generating nanobody CDR3 sequences. This repository includes pretrained weights, generated libraries, predicted structures, and training and evaluation scripts.

[Installation](#installation) · [Quick start](#quick-start) · [Libraries](#provided-libraries) · [Structure prediction](#structure-prediction) · [Evaluation](#evaluation) · [Training and PPL](#training-and-ppl) · [Support](#support)

## Installation

Generation requires Python 3.9 and an NVIDIA GPU. From the repository root on Linux with Conda, install the main environment:

```bash
conda env create -f environment.yml
conda activate vhh-llama
```

This environment installs [requirements.txt](requirements.txt) and supports generation, training, PPL and property evaluation. Structure prediction uses a separate environment below.

## Quick start

Use the pretrained model in `weights/VHH-Llama/` to generate a small CDR3 library for the three example scaffolds in `examples/therapeutic_vhh_scaffolds.csv`.

In `configs/gen.yaml`, set `samples_return_per_input: 10` and `device: [0]` under `generate`, keeping the other settings. Run:

```bash
python src/Generate.py --config configs/gen.yaml
```

The first run saves 10 unique CDR3s per scaffold (30 in total) to `outputs/libs/gen_VHH-Llama(1).csv`, with configuration and statistics files. Later runs use a new batch number.

<details>
<summary>Generation settings, custom scaffolds, and CSV output</summary>

| Setting in `configs/gen.yaml` | Purpose |
|---|---|
| `data.input_csv` | Scaffold CSV; the supplied file has three examples |
| `generate.samples_return_per_input` | Unique CDR3s per scaffold; supplied value: `100000` |
| `generate.sampling_batch_size` | Sequences sampled per GPU round |
| `generate.temperature`, `generate.top_p` | Sampling parameters |
| `generate.device` | GPU index, e.g. `[0]`; one GPU per run |

Custom scaffold CSVs require `L_seq` (VHH sequence before CDR3) and `R_seq` (sequence after CDR3); `name` is optional.

Generated CSV columns:

```text
row_index,sample_index,generated_cdr3,length,eos_reached,name,L_seq,R_seq
```

`row_index` identifies the zero-based input scaffold; `sample_index` identifies a CDR3 within that scaffold. Uniqueness is enforced per scaffold. Reconstruct a full VHH as `L_seq + generated_cdr3 + R_seq`.

</details>

## Provided libraries

| Library | Generated CDR3s | Sampled structure inputs | Available PDBs |
|---|---:|---:|---:|
| IgLM | 300,000 | 30,000 | 30,000 |
| VHH-Llama | 300,000 | 30,000 | 29,998 |

Each `VHH_libraries/<library>/` contains `generated_cdr3.csv`, `structure_prediction_input.csv`, and `predicted_structures/`. Existing PDBs can be used directly for evaluation.

## Structure prediction

Create the separate NanoBodyBuilder2 environment and predict a generated batch:

```bash
conda env create -f environment_structure.yml
conda activate vhh-llama-structure
python evaluation/structure_eval/predict_structures.py \
  'outputs/libs/gen_VHH-Llama(1).csv' --gpus 0 --workers-per-gpu 1
```

Output: `outputs/predicted_structures/gen_VHH-Llama(1)/row_<n>/`. ImmuneBuilder downloads missing model weights on first use; the environment requires a CUDA 11.8-compatible NVIDIA driver.

<details>
<summary>Other inputs, reproduction, and prediction options</summary>

- Complete VHH sequences: use a CSV with `VHH_seq` or `VHH_sequence`.
- CDR3-only files: require `row_index`, `sample_index`, `generated_cdr3`, plus inline `L_seq`/`R_seq` or a scaffold CSV supplied with `--frameworks`.
- Article reproduction: use `VHH_libraries/<library>/structure_prediction_input.csv`; these files automatically use the example scaffolds. A subset saved elsewhere needs `--frameworks examples/therapeutic_vhh_scaffolds.csv`.
- Options: `--gpus` selects GPUs, `--workers-per-gpu` sets processes, and `--row-indices` selects scaffolds. Re-running resumes existing results; use `--overwrite` or `--retry-failed` as needed.

Ordinary CSVs use their filename stem as the output folder; `structure_prediction_input.csv` uses its parent library name. Multiple CSV inputs are supported.

Existing PDB layout: `VHH_libraries/<library>/predicted_structures/row_<n>/row_<n>_sample_<index>.pdb`. The two VHH-Llama failures are recorded in `VHH_libraries/VHH-Llama/predicted_structures/row_1_failures.json`.

</details>

## Evaluation

Each evaluator processes one library and writes CSV values. For example, calculate GRAVY for a generated batch:

```bash
conda activate vhh-llama
python evaluation/developability/Hydrophobicity/GRAVY.py \
  --input 'outputs/libs/gen_VHH-Llama(1).csv'
```

| Metric | Input | Script |
|---|---|---|
| Net charge at pH 7.4 | VHH sequence | [Charge.py](evaluation/developability/Electrostatics/Charge.py) |
| GRAVY | CDR3 sequence | [GRAVY.py](evaluation/developability/Hydrophobicity/GRAVY.py) |
| OASis identity | VHH sequence | [OASis_identity.py](evaluation/developability/Humanness/OASis_identity.py) |
| CDR3 SAP | PDB structures | [SAP.py](evaluation/developability/Aggregation_propensity/SAP.py) |
| PPC/PNC | PDB structures | [PPC_PNC.py](evaluation/developability/Electrostatics/PPC_PNC.py) |
| SASA and hydrophobic fraction | PDB structures | [SASA.py](evaluation/developability/Hydrophobicity/SASA.py) |
| CDR3–FR2 contact score | PDB structures | [cdr3_fr2_contact.py](evaluation/structure_eval/cdr3_fr2_contact.py) |

Results are saved under `outputs/developability/<library-or-batch>/` or, for contacts, `outputs/structure_eval/<library-or-batch>/`.

<details>
<summary>Evaluation inputs, options, and external resources</summary>

Sequence evaluators accept the generated CSV format above; older CDR3-only inputs need `--frameworks`. Structure evaluators accept a directory following the PDB layout above. SAP, PPC/PNC and contact scoring require IMGT-numbered chain H.

Use `--input` to select a library, `--output` to choose a CSV, `--limit` for a subset, and `--workers` for CPU processes in structure evaluation. Without `--input`, the provided VHH-Llama library is used. Inspect result status and error columns.

**OASis:** download and decompress the official [9-mer database](https://zenodo.org/record/5164685) (~23 GB). It is not included in this repository or the VHH-Llama Hugging Face dataset.

```bash
mkdir -p datasets
wget -O datasets/OASis_9mers_v1.db.gz https://zenodo.org/record/5164685/files/OASis_9mers_v1.db.gz
gunzip datasets/OASis_9mers_v1.db.gz
```

The default database path is `datasets/OASis_9mers_v1.db`; use `--oasis-db /path/to/OASis_9mers_v1.db` for an existing copy.

**PPC/PNC:** uses [TNP](https://github.com/oxpig/TNP) and its bundled PSA executable, installed through `requirements.txt`.

**SAP/SASA:** PyRosetta is included in the main environment; its official wheel source is configured in `requirements.txt`.

</details>

## Training and PPL

Download the CSVs from [Hugging Face](https://huggingface.co/datasets/Parken/VHH-Llama) and save them at these paths:

| File | Records | Purpose |
|---|---:|---|
| `datasets/train.csv` | 16,964,278 | Training |
| `datasets/val.csv` | 171,356 | Validation |
| `datasets/test.csv` | 1,979,175 | VHHCorpus PPL evaluation |

Review GPU and training settings in `configs/train.yaml`, then train:

```bash
conda activate vhh-llama
python src/Train.py --config configs/train.yaml
```

Calculate PPL with the provided model:

```bash
conda activate vhh-llama
python evaluation/ppl/PPL.py --checkpoint weights/VHH-Llama --input_csv datasets/test.csv
```

Training outputs: `outputs/VHH-Llama/versionN/`. PPL results: `outputs/PPL/`.

<details>
<summary>Dataset format and PPL options</summary>

Training and validation require this exact header and column order:

```text
CDR3_seq,VHH_seq,L_seq,R_seq
```

PPL requires `L_seq`, `CDR3_seq` and `R_seq`. Use `--limit 10` for a small run or `--visible_gpus 0,1` for multiple GPUs.

</details>

## Support

Please open a GitHub issue with the command, configuration and error log.
