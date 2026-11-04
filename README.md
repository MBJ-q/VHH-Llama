# VHH-Llama

VHH-Llama is a language model for generating nanobody CDR3 sequences. This repository includes pretrained weights, generated libraries, predicted structures, and training and evaluation scripts.

## Installation

Generation requires Python 3.9 and an NVIDIA GPU. From the repository root on Linux with Conda, install the main environment:

```bash
conda env create -f environment.yml
conda activate vhh-llama
```

This environment installs [requirements.txt](requirements.txt) and supports generation, training, PPL and property evaluation. Structure prediction uses a separate environment below.

## Library generation

You can adjust `temperature`, `top_p`, and other generation parameters in `configs/gen.yaml` to generate your library.

Run from the repository root using [configs/gen.yaml](configs/gen.yaml):

```bash
conda activate vhh-llama
python src/Generate.py --config configs/gen.yaml
```

Output: `outputs/libs/gen_VHH-Llama(<batch>).csv`, with configuration and statistics files. Each run uses a new batch number.

## Provided libraries

| Library | Generated CDR3s | Sampled structure inputs | Available PDBs |
|---|---:|---:|---:|
| IgLM | 300,000 | 30,000 | 30,000 |
| VHH-Llama | 300,000 | 30,000 | 29,998 |

Each `VHH_libraries/<library>/` contains `generated_cdr3.csv`, `structure_prediction_input.csv`, and `predicted_structures/`. Existing PDBs can be used directly for evaluation.

## Structure prediction

Use NanoBodyBuilder2 to predict VHH structures from your generated library. Create the separate structure prediction environment, then run the prediction script:

```bash
conda env create -f environment_structure.yml
conda activate vhh-llama-structure
python evaluation/structure_eval/predict_structures.py \
  'outputs/libs/gen_VHH-Llama(1).csv' --gpus 0 --workers-per-gpu 1
```

Replace the CSV path with your generated library and set `--gpus` to the GPU you want to use. For this example, predicted PDB files are saved under `outputs/predicted_structures/gen_VHH-Llama(1)/`.

The environment requires a CUDA 11.8-compatible NVIDIA driver. ImmuneBuilder downloads model weights on first use.

## Evaluation

Use your generated CSV for sequence evaluation and the predicted PDBs for structure evaluation. Run the evaluators in the main environment:

```bash
conda activate vhh-llama
```

For example, calculate CDR3 GRAVY from a generated library:

```bash
python evaluation/developability/Hydrophobicity/GRAVY.py \
  --input 'outputs/libs/gen_VHH-Llama(1).csv'
```

Calculate CDR3 SAP from its predicted structures:

```bash
python evaluation/developability/Aggregation_propensity/SAP.py \
  --input 'outputs/predicted_structures/gen_VHH-Llama(1)'
```

Replace these example paths with your own CSV or structure directory. You can also evaluate the supplied libraries using `VHH_libraries/<library>/generated_cdr3.csv` or `VHH_libraries/<library>/predicted_structures/`.

| Metric | Input | Script |
|---|---|---|
| Net charge at pH 7.4 | Generated CSV | [Charge.py](evaluation/developability/Electrostatics/Charge.py) |
| GRAVY | Generated CSV | [GRAVY.py](evaluation/developability/Hydrophobicity/GRAVY.py) |
| OASis identity | Generated CSV | [OASis_identity.py](evaluation/developability/Humanness/OASis_identity.py) |
| CDR3 SAP | PDB structures | [SAP.py](evaluation/developability/Aggregation_propensity/SAP.py) |
| PPC/PNC | PDB structures | [PPC_PNC.py](evaluation/developability/Electrostatics/PPC_PNC.py) |
| SASA and hydrophobic fraction | PDB structures | [SASA.py](evaluation/developability/Hydrophobicity/SASA.py) |
| CDR3–FR2 contact score | PDB structures | [cdr3_fr2_contact.py](evaluation/structure_eval/cdr3_fr2_contact.py) |

For the example batch, evaluation CSVs are saved under `outputs/developability/gen_VHH-Llama(1)/`; CDR3–FR2 contact results are saved under `outputs/structure_eval/gen_VHH-Llama(1)/`.

SAP, PPC/PNC, and contact scoring require IMGT-numbered chain H in the PDB files.

<details>
<summary>Evaluation dependencies</summary>

**OASis:** download and extract the [9-mer database](https://zenodo.org/record/5164685) to the default path `datasets/OASis_9mers_v1.db`, or specify another path with `--oasis-db /path/to/OASis_9mers_v1.db`.

**PPC/PNC:** uses [TNP](https://github.com/oxpig/TNP) and its bundled PSA executable, installed through `requirements.txt`.

**SAP/SASA:** PyRosetta is included in the main environment; its official wheel source is configured in `requirements.txt`.

</details>

## Training and PPL

Download the CSVs from [Hugging Face](https://huggingface.co/datasets/Parken/VHH-Llama) and save them at these paths:

| File | Purpose |
|---|---|
| `datasets/train.csv` | Training |
| `datasets/val.csv` | Validation |
| `datasets/test.csv` | VHHCorpus PPL evaluation |

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
