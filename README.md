# VHH-Llama

VHH-Llama generates nanobody (VHH) CDR3 sequences conditioned on user-provided VHH sequence contexts. This repository provides pretrained weights, generated sequence libraries, predicted structures, and scripts for training, generation, structure prediction, and evaluation.

[Installation](#installation) · [Quick start](#quick-start) · [Generation](#generate-cdr3-sequences) · [Structures](#predict-or-use-vhh-structures) · [Evaluation](#evaluate-sequences-and-structures) · [Training and PPL](#training-and-perplexity)

## Repository structure

```text
.
├── weights/VHH-Llama/       # Pretrained weights, model config and vocabulary
├── src/                    # Model, data loading, training and generation
├── configs/                # Training and generation settings
├── examples/               # Example VHH sequence contexts
├── VHH_libraries/
│   ├── IgLM/               # Generated CDR3s, sampled inputs and PDBs
│   └── VHH-Llama/
├── evaluation/
│   ├── ppl/                # CDR3 perplexity
│   ├── developability/     # Sequence and structure descriptors
│   └── structure_eval/     # Structure prediction and CDR3–FR2 contacts
├── datasets/               # Downloaded datasets and external databases
├── outputs/                # Results created by running the scripts
├── environment.yml         # Main environment
├── environment_structure.yml
├── requirements.txt
└── DEPENDENCIES.txt         # Detailed dependency and resource preparation
```

`datasets/` and `outputs/` are local working directories excluded from Git.

## Included resources

| Library | Generated CDR3s | Sampled structure inputs | Available PDBs |
|---|---:|---:|---:|
| IgLM | 300,000 | 30,000 | 30,000 |
| VHH-Llama | 300,000 | 30,000 | 29,998 |

Each library contains:

- `generated_cdr3.csv`: the generated sequence library.
- `structure_prediction_input.csv`: the sampled sequences used for structure prediction.
- `predicted_structures/`: the available predicted PDBs.

The libraries cover the three contexts in `examples/therapeutic_vhh_scaffolds.csv`. The two failed VHH-Llama predictions are recorded in `predicted_structures/row_1_failures.json`.

Pretrained model files are provided in `weights/VHH-Llama/`.

## Installation

Run the commands below from the repository root. The installation instructions target Linux with Conda.

| Environment | Used for |
|---|---|
| `environment.yml` — Python 3.9 | Training, generation, PPL and property evaluation |
| `environment_structure.yml` — Python 3.10 | NanoBodyBuilder2 structure prediction |

Create the main environment:

```bash
conda env create -f environment.yml
conda activate vhh-llama
```

Generation requires an NVIDIA GPU. Structure prediction uses a separate CUDA 11.8 environment described below.

Additional resources are needed for specific metrics:

- **OASis identity:** download the OASis database separately, approximately 22 GB decompressed.
- **PPC/PNC:** the PSA executable distributed with TNP must be executable and available on `PATH`.
- **SAP and SASA:** require PyRosetta, included in the main dependency list.

See [DEPENDENCIES.txt](DEPENDENCIES.txt) for preparation commands and official sources.

## Quick start

Calculate CDR3 GRAVY for the first 10 sequences in the provided VHH-Llama library:

```bash
python evaluation/developability/Hydrophobicity/GRAVY.py --limit 10
```

Output:

```text
outputs/developability/VHH-Llama/gravy_values.csv
```

This example uses the provided sequences and does not require structure prediction or a GPU.

## Generate CDR3 sequences

Edit `configs/gen.yaml` to select the input CSV, GPU and sampling settings. For a small first run, set `generate.samples_return_per_input` to `10` and `generate.device` to `[0]`.

```bash
python src/Generate.py --config configs/gen.yaml
```

The default input is `examples/therapeutic_vhh_scaffolds.csv`. To use custom VHH contexts, provide a CSV containing `L_seq` and `R_seq`; `name` is optional.

The complete generated VHH is reconstructed as:

```text
L_seq + generated_cdr3 + R_seq
```

Results are saved as separate batches:

```text
outputs/libs/gen_VHH-Llama(1).csv
outputs/libs/gen_VHH-Llama(1)_config.yaml
outputs/libs/gen_VHH-Llama(1)_stats.json
```

Generated CSVs retain the sequence contexts and sample identifiers for downstream evaluation and structure prediction.

The supplied configuration generates 100,000 unique CDR3s per context. Generation currently supports one GPU per run.

## Predict or use VHH structures

### Use the provided PDBs

Existing structures can be used directly for evaluation:

```text
VHH_libraries/IgLM/predicted_structures/
VHH_libraries/VHH-Llama/predicted_structures/
```

A structure is identified by its library, `row_index` and `sample_index`:

```text
row_<row_index>/row_<row_index>_sample_<sample_index>.pdb
```

### Predict new structures

Create and activate the structure prediction environment:

```bash
conda env create -f environment_structure.yml
conda activate vhh-llama-structure
```

Predict a generated batch:

```bash
python evaluation/structure_eval/predict_structures.py \
  'outputs/libs/gen_VHH-Llama(1).csv' \
  --gpus 0 --workers-per-gpu 1
```

Output:

```text
outputs/predicted_structures/gen_VHH-Llama(1)/row_<n>/
```

The predictor also accepts custom CSVs containing complete sequences in a `VHH_seq` or `VHH_sequence` column. Older CDR3-only CSVs require a context CSV supplied through `--frameworks`.

To reproduce the sampled IgLM structure predictions:

```bash
python evaluation/structure_eval/predict_structures.py \
  VHH_libraries/IgLM/structure_prediction_input.csv \
  --gpus 0 --workers-per-gpu 1
```

Use the corresponding VHH-Llama CSV for that library. Each sampled CSV contains 30,000 sequences.

GPU indices and workers per GPU are configurable. Re-running the same input resumes from existing results; `--overwrite` predicts again and `--retry-failed` retries recorded failures. ImmuneBuilder downloads missing NanoBodyBuilder2 weights when the predictor is initialized.

See [DEPENDENCIES.txt](DEPENDENCIES.txt) for input formats, subset prediction and output naming.

## Evaluate sequences and structures

Activate the main environment:

```bash
conda activate vhh-llama
```

Each script analyzes one library and writes calculated values to CSV.

| Metric | Input and calculation | Script |
|---|---|---|
| Charge | Full-VHH net charge at pH 7.4 | [Charge.py](evaluation/developability/Electrostatics/Charge.py) |
| GRAVY | CDR3 hydropathy | [GRAVY.py](evaluation/developability/Hydrophobicity/GRAVY.py) |
| OASis identity | Full-VHH human 9-mer identity at 50% prevalence | [OASis_identity.py](evaluation/developability/Humanness/OASis_identity.py) |
| SAP | Structure-based CDR3 SAP and SAP per residue | [SAP.py](evaluation/developability/Aggregation_propensity/SAP.py) |
| PPC/PNC | Positive and negative CDR charge patches | [PPC_PNC.py](evaluation/developability/Electrostatics/PPC_PNC.py) |
| SASA | Total and hydrophobic SASA, including hydrophobic fraction | [SASA.py](evaluation/developability/Hydrophobicity/SASA.py) |
| CDR3–FR2 contact | Soft contact score from IMGT-numbered structures | [cdr3_fr2_contact.py](evaluation/structure_eval/cdr3_fr2_contact.py) |

Defaults use the provided VHH-Llama sequence library or structures. Select another library with `--input`, change the output with `--output`, and use `--limit` for a small run. Structure evaluators support CPU processes through `--workers`.

For example, evaluate 10 existing IgLM structures:

```bash
python evaluation/developability/Hydrophobicity/SASA.py \
  --input VHH_libraries/IgLM/predicted_structures --limit 10
```

Calculate CDR3–FR2 contacts:

```bash
python evaluation/structure_eval/cdr3_fr2_contact.py --limit 10
```

Default output locations:

```text
outputs/developability/<library-or-batch>/<metric>_values.csv
outputs/structure_eval/<library-or-batch>/cdr3_fr2_contact_values.csv
```

Sequence evaluators read the generated CDR3 and its left/right contexts. Custom inputs require `row_index`, `sample_index`, `generated_cdr3`, and either inline `L_seq`/`R_seq` or `--frameworks`.

Structure inputs must follow the PDB directory naming shown above. SAP, PPC/PNC and CDR3–FR2 contact calculations require IMGT numbering on chain H; these evaluators do not renumber structures. Inspect the status and error columns when using evaluation results.

## Training and perplexity

### Download the datasets

Obtain the training, validation and PPL evaluation CSVs from the [Hugging Face dataset repository](https://huggingface.co/datasets/Parken/VHH-Llama), and save them as:

```text
datasets/train.csv
datasets/val.csv
datasets/test.csv
```

| File | Records | Purpose |
|---|---:|---|
| `train.csv` | 16,964,278 | Model training |
| `val.csv` | 171,356 | Validation and checkpoint selection |
| `test.csv` | 1,979,175 | VHHCorpus PPL evaluation |

Training and validation are split by complete VHH Linclust clusters at the recorded 85% clustering threshold, using seed 42 and approximately 1% of sequences for validation. The PPL evaluation set comes from VHHCorpus and is separate from this train/validation partition.

Training and validation CSVs require this exact header and column order:

```text
CDR3_seq,VHH_seq,L_seq,R_seq
```

PPL inputs require `L_seq`, `CDR3_seq` and `R_seq`. Optional subject metadata columns are preserved in the results.

### Train a model

Review the GPU and training settings in `configs/train.yaml`, then run:

```bash
python src/Train.py --config configs/train.yaml
```

Training runs are saved under `outputs/VHH-Llama/versionN/`, including checkpoints, the best model, the final model, configuration snapshots and training results.

### Calculate PPL

Evaluate the provided pretrained model:

```bash
python evaluation/ppl/PPL.py \
  --checkpoint weights/VHH-Llama \
  --input_csv datasets/test.csv
```

Results are written under `outputs/PPL/` and include per-position CDR3 probabilities and perplexity summaries with and without EOS. Use `--limit 10` for a small run; `--visible_gpus` enables GPU-based data sharding.

## Support

Please open a GitHub issue with the command, configuration and error log when reporting a problem.

Detailed dependency installation, external resources and structure workflows are documented in [DEPENDENCIES.txt](DEPENDENCIES.txt).
