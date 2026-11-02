# VHH-Llama

VHH-Llama is a language model for generating nanobody CDR3 sequences. This repository includes pretrained weights, generated libraries, predicted structures, and training and evaluation scripts.

[Installation](#installation) · [Quick start](#quick-start) · [Generation](#generate-cdr3-sequences) · [Structures](#predict-or-use-vhh-structures) · [Evaluation](#evaluate-sequences-and-structures) · [Training and PPL](#training-and-perplexity) · [Support](#support)

## Installation

Run commands from the repository root. These instructions target Linux with Conda; generation and structure prediction require an NVIDIA GPU.

| Environment | Used for |
|---|---|
| `environment.yml` — Python 3.9 | Training, generation, PPL and property evaluation |
| `environment_structure.yml` — Python 3.10 | NanoBodyBuilder2 structure prediction |

```bash
conda env create -f environment.yml
conda activate vhh-llama
```

Users must obtain the OASis database separately from its official source (~23 GB decompressed); it is not included in this repository or the Hugging Face dataset. PPC/PNC requires TNP's PSA executable on `PATH`. SAP and SASA use PyRosetta, included in the main dependencies.

See [DEPENDENCIES.txt](DEPENDENCIES.txt) for installation details and resource preparation.

## Quick start

Calculate GRAVY for 10 provided VHH-Llama CDR3s on CPU:

```bash
python evaluation/developability/Hydrophobicity/GRAVY.py --limit 10
```

Output: `outputs/developability/VHH-Llama/gravy_values.csv`.

## Generate CDR3 sequences

Pretrained weights are in `weights/VHH-Llama/`. Configure the input, GPU and sampling settings in `configs/gen.yaml`. For a small first run, set `generate.samples_return_per_input` to `10` and `generate.device` to `[0]`.

```bash
python src/Generate.py --config configs/gen.yaml
```

The default input is `examples/therapeutic_vhh_scaffolds.csv`. Custom context CSVs require `L_seq` and `R_seq`; `name` is optional. Full VHH sequences are reconstructed as `L_seq + generated_cdr3 + R_seq`.

Generated batches are saved as `outputs/libs/gen_VHH-Llama(1).csv`, with matching configuration and statistics files. Each CSV retains the sequence contexts and sample identifiers for downstream analysis.

The default configuration generates 100,000 unique CDR3s per context. Generation currently supports one GPU per run.

## Predict or use VHH structures

### Provided libraries and structures

| Library | Generated CDR3s | Sampled structure inputs | Available PDBs |
|---|---:|---:|---:|
| IgLM | 300,000 | 30,000 | 30,000 |
| VHH-Llama | 300,000 | 30,000 | 29,998 |

Each `VHH_libraries/<library>/` contains `generated_cdr3.csv`, `structure_prediction_input.csv`, and `predicted_structures/`. The libraries use the three contexts in `examples/therapeutic_vhh_scaffolds.csv`.

Existing PDBs can be evaluated directly, using the main environment. Locate them by `row_index` and `sample_index`:

```text
VHH_libraries/<library>/predicted_structures/row_<n>/row_<n>_sample_<index>.pdb
```

The two failed VHH-Llama predictions are recorded in `predicted_structures/row_1_failures.json`.

### Predict new structures

```bash
conda env create -f environment_structure.yml
conda activate vhh-llama-structure
```

Predict a generated batch:

```bash
python evaluation/structure_eval/predict_structures.py 'outputs/libs/gen_VHH-Llama(1).csv' \
  --gpus 0 --workers-per-gpu 1
```

Output: `outputs/predicted_structures/gen_VHH-Llama(1)/row_<n>/`.

For custom full VHH sequences, supply a CSV with `VHH_seq` or `VHH_sequence`. CDR3-only inputs require inline `L_seq`/`R_seq` or an external context CSV through `--frameworks`.

To reproduce sampled predictions, replace the input above with `VHH_libraries/<library>/structure_prediction_input.csv`. Each sampled CSV contains 30,000 sequences; omitting input arguments selects both provided libraries.

GPU indices and workers are configurable. ImmuneBuilder downloads missing NanoBodyBuilder2 weights on first use. Resume, subset and retry options are documented in [DEPENDENCIES.txt](DEPENDENCIES.txt).

## Evaluate sequences and structures

Switch back to the main environment:

```bash
conda activate vhh-llama
```

Each script analyzes one library and outputs CSV values.

| Metric | Input and calculation | Script |
|---|---|---|
| Charge | Full-VHH net charge at pH 7.4 | [Charge.py](evaluation/developability/Electrostatics/Charge.py) |
| GRAVY | CDR3 hydropathy | [GRAVY.py](evaluation/developability/Hydrophobicity/GRAVY.py) |
| OASis identity | Full-VHH human 9-mer identity at 50% prevalence | [OASis_identity.py](evaluation/developability/Humanness/OASis_identity.py) |
| SAP | Structure-based CDR3 SAP and SAP per residue | [SAP.py](evaluation/developability/Aggregation_propensity/SAP.py) |
| PPC/PNC | Positive and negative CDR charge patches | [PPC_PNC.py](evaluation/developability/Electrostatics/PPC_PNC.py) |
| SASA | Total and hydrophobic SASA, including hydrophobic fraction | [SASA.py](evaluation/developability/Hydrophobicity/SASA.py) |
| CDR3–FR2 contact | Structure-based soft contact score | [cdr3_fr2_contact.py](evaluation/structure_eval/cdr3_fr2_contact.py) |

Defaults use the provided VHH-Llama sequences or structures. Use `--input` for another library, `--output` for another CSV, and `--limit` for a small run. Structure evaluators also accept `--workers`.

For example, evaluate 10 existing IgLM structures:

```bash
python evaluation/developability/Hydrophobicity/SASA.py \
  --input VHH_libraries/IgLM/predicted_structures --limit 10
```

Calculate CDR3–FR2 contacts:

```bash
python evaluation/structure_eval/cdr3_fr2_contact.py --limit 10
```

Default outputs:

```text
outputs/developability/<library-or-batch>/<metric>_values.csv
outputs/structure_eval/<library-or-batch>/cdr3_fr2_contact_values.csv
```

Custom sequence inputs require `row_index`, `sample_index`, `generated_cdr3`, and inline `L_seq`/`R_seq` or `--frameworks`.

Structure inputs follow the PDB naming above. SAP, PPC/PNC and CDR3–FR2 contacts require IMGT numbering on chain H. Inspect the status and error columns before using results.

## Training and perplexity

### Download the datasets

Download the CSVs from [Hugging Face](https://huggingface.co/datasets/Parken/VHH-Llama) and save them at the following paths:

| File | Records | Purpose |
|---|---:|---|
| `datasets/train.csv` | 16,964,278 | Training |
| `datasets/val.csv` | 171,356 | Validation and checkpoint selection |
| `datasets/test.csv` | 1,979,175 | VHHCorpus PPL evaluation |

Train/validation splitting uses complete VHH Linclust clusters at the recorded 85% threshold, seed 42, and approximately 1% of sequences for validation. The VHHCorpus PPL set is separate from this partition.

Training and validation require this exact header and column order:

```text
CDR3_seq,VHH_seq,L_seq,R_seq
```

PPL inputs require `L_seq`, `CDR3_seq` and `R_seq`.

### Train a model

Review GPU and training settings in `configs/train.yaml`, then run:

```bash
python src/Train.py --config configs/train.yaml
```

Checkpoints, best/final models and training results are saved under `outputs/VHH-Llama/versionN/`.

### Calculate PPL

```bash
python evaluation/ppl/PPL.py \
  --checkpoint weights/VHH-Llama \
  --input_csv datasets/test.csv
```

Results in `outputs/PPL/` include per-position CDR3 probabilities and perplexity summaries with and without EOS. Use `--limit 10` for a small run.

## Support

Please open a GitHub issue with the command, configuration and error log.
