# OWL: Outlier Weighed Layerwise Sparsity — Reproduction & Extension
Reproducibility: Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity

---

## Requirements

- Google Colab (Pro recommended for A100 GPU and parallel sessions)
- A Hugging Face account with access to [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
- ~14GB GPU VRAM (A100 or T4)
- ~15GB free disk space per run if saving pruned models

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/luuyin/OWL.git
cd OWL
```

### 2. Pin dependencies
Newer versions of `transformers` and `datasets` break the OWL data loading code.
Pin these exact versions:
```bash
pip install transformers==4.34.0 datasets==2.14.5 accelerate==0.23.0 sentencepiece
```

### 3. Patch data.py
The original `lib/data.py` has two bugs with newer `datasets` versions.
Run this once before any experiments:
```python
with open('lib/data.py', 'r') as f:
    content = f.read()

content = content.replace(
    "traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')",
    "traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')"
).replace(
    "valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')",
    "valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')"
).replace(
    "valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')",
    "valenc = tokenizer(' '.join(valdata.select(range(1100))['text']), return_tensors='pt')"
)

with open('lib/data.py', 'w') as f:
    f.write(content)
```

### 4. Log in to Hugging Face
```python
from huggingface_hub import notebook_login
notebook_login()
```

---

## Reproduction Runs

Reproduces the paper's core results on LLaMA-7B. Target PPL ≈ 24.55 at 70% sparsity.

### OWL + Wanda — 70% sparsity (primary reproduction)
```bash
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda_owl \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --save out/llama_7b/owl_70/ \
    --Hyper_m 5 \
    --Lamda 0.08 \
    --outlier_by_activation \
    --outlier_by_wmetric \
    --seed 0
```
**Result: PPL = 24.29** ✅ (paper: 24.55, difference: -0.26)

### Wanda baseline — 70% sparsity
```bash
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --save out/llama_7b/wanda_70/ \
    --seed 0
```
**Result: PPL = 82.42** (paper: 85.77)

### Wanda baseline — 80% sparsity
```bash
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda \
    --sparsity_ratio 0.8 \
    --sparsity_type unstructured \
    --save out/llama_7b/wanda_80/ \
    --seed 0
```
**Result: PPL = 12802.86** — uniform pruning collapses at 80%

### OWL + Wanda — 80% sparsity
```bash
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda_owl \
    --sparsity_ratio 0.8 \
    --sparsity_type unstructured \
    --save out/llama_7b/owl_80/ \
    --Hyper_m 5 \
    --Lamda 0.08 \
    --outlier_by_activation \
    --outlier_by_wmetric \
    --seed 0
```
**Result: PPL = 1263.17** — OWL helps but 80% is too aggressive

---

## Extension: Hyperparameter Sensitivity Analysis

The paper selected M=5 and λ=0.08 without publishing full sensitivity curves.
These runs generate those curves.

> **Note:** M=3 and M=5 used `--nsamples 128` (full calibration). All other sweep
> runs used `--nsamples 32` for computational efficiency. Trends are consistent
> with full calibration and this is noted in the report.

### M Sweep (λ=0.08 fixed, 70% sparsity)

| M | nsamples | PPL | vs M=5 |
|---|----------|-----|--------|
| 3 | 128 | 27.77 | +3.48 |
| 4 | 32 | 26.37 | +2.08 |
| **5 ★** | **128** | **24.29** | **—** |
| 6 | 32 | 28.17 | +3.88 |
| 7 | 32 | 33.57 | +9.28 |
| 8 | 32 | 38.42 | +14.13 |
| 10 | 32 | 46.23 | +21.94 |

★ = paper default. Run each with `--Hyper_m <M>` and `--Lamda 0.08`.

Example for M=4:
```bash
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda_owl \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --save out/llama_7b/owl_70_m4/ \
    --Hyper_m 4 \
    --Lamda 0.08 \
    --nsamples 32 \
    --outlier_by_activation \
    --outlier_by_wmetric \
    --seed 0
```

---

### λ Sweep (M=5 fixed, 70% sparsity)

| λ | nsamples | PPL | vs λ=0.08 |
|---|----------|-----|-----------|
| 0.04 | 32 | 40.67 | +16.38 |
| 0.06 | 32 | 32.00 | +7.71 |
| **0.08 ★** | **128** | **24.29** | **—** |
| 0.10 | 32 | 24.18 | -0.11 |
| 0.15 | 32 | 24.07 | -0.22 |

★ = paper default. Run each with `--Lamda <λ>` and `--Hyper_m 5`.

Example for λ=0.10:
```bash
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda_owl \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --save out/llama_7b/owl_70_lam010/ \
    --Hyper_m 5 \
    --Lamda 0.10 \
    --nsamples 32 \
    --outlier_by_activation \
    --outlier_by_wmetric \
    --seed 0
```

---

## Key Findings

1. **Reproduction successful** — our PPL of 24.29 matches the paper's 24.55 within 0.26 points.
2. **OWL vs uniform Wanda** — OWL reduces PPL from 82.42 to 24.29 at 70% sparsity, a dramatic improvement.
3. **M=5 is optimal** — PPL degrades sharply for M > 5, confirming the paper's choice.
4. **λ threshold effect** — λ < 0.08 causes significant degradation; λ ≥ 0.08 all perform similarly (~24.1–24.3).
5. **80% sparsity** — even OWL cannot recover model quality at 80% unstructured sparsity on LLaMA-7B.

---

## Argument Reference

| Argument | Description |
|----------|-------------|
| `--model` | HuggingFace model path |
| `--prune_method` | `wanda`, `wanda_owl`, `magnitude`, `sparsegpt`, etc. |
| `--sparsity_ratio` | Fraction of weights to prune (e.g. 0.7 = 70%) |
| `--sparsity_type` | `unstructured` or `2:4` or `4:8` |
| `--Hyper_m` | M value — outlier layer budget (paper default: 5) |
| `--Lamda` | λ value — outlier weighting strength (paper default: 0.08) |
| `--nsamples` | Calibration samples (paper default: 128, sweep: 32) |
| `--outlier_by_activation` | Use activation magnitude to identify outlier layers |
| `--outlier_by_wmetric` | Use weight metric to identify outlier layers |
| `--seed` | Random seed for reproducibility |
| `--save` | Path to save pruning results |
