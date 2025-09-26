# SynCED-EnDe — Pipeline, Raw Data & Scripts (GitHub Companion)

This repository provides **pipeline scripts and raw/intermediate files** for building the [SynCED-EnDe dataset](https://huggingface.co/datasets/moon712/SynCED_EnDe_2025).

The **clean, benchmark-ready release** (train silver + eval gold) is on Hugging Face.\
This repo is for **reproducibility and transparency**.

---

## 📂 Repository Structure

```
scripts/                 # Core dataset creation pipeline
  labels.py                 # compute label distributions and schema
  inject.py                 # inject controlled errors
  reinject.py               # retry injection for missing cases
  skim.py                   # preview dataset (sanity check)
  block_evalrows.py         # filter out evaluation rows
  final.py                  # assemble final dataset TSVs
  data_check.py             # sanity checks (counts, distributions, label balance)
  data_scrape.py            # scrape GOV.UK + Stack Exchange (FDA removed)

data/
  train/
    synced_ende_train_silver.tsv     # final train split
    raw/                             # raw/intermediate training data
      ced_final_injected_multi_class.tsv
      error_injected_rows_with_correctMT.tsv
      rows_for_error_injection.tsv
      sources_2024_2025.tsv

  eval/
    synced_ende_eval_gold.tsv        # final eval split
    eval_judged_quantified_annotated.tsv
    raw/                             # raw/intermediate eval data
      ced_final_injected_multi_class.tsv
      error_injected_rows_with_correctMT.tsv
      rows_for_error_injection.tsv
      sources_2024_2025.tsv
```

---

## 🛠️ Pipeline Usage

Run the scripts in **order**:

1. **Define labels**

   ```bash
   python scripts/labels.py
   ```

2. **Inject errors (first pass)**

   ```bash
   python scripts/inject.py
   ```

3. **Re-inject (fix empty rows, optional second pass)**

   ```bash
   python scripts/reinject.py
   ```

4. **Preview dataset (skim head rows)**

   ```bash
   python scripts/skim.py
   ```

5. **Block evaluation rows (avoid leakage)**

   ```bash
   python scripts/block_evalrows.py
   ```

6. **Assemble final dataset**

   ```bash
   python scripts/final.py
   ```

   This produces:

   - `synced_ende_train_silver.tsv`
   - `synced_ende_eval_gold.tsv`
   - `judged_quantified_annotated.tsv`

---

## 📊 Raw / Intermediate Data

Available in `data/`:

- `ced_final_injected_multi_class.tsv`
- `error_injected_rows_with_correctMT.tsv`
- `rows_for_error_injection.tsv`
- `sources_2024_2025.tsv`

⚠️ **Note:** These files may contain earlier/unverified labels.\
They are **not for benchmarking** — use only for reproducing pipeline steps.

---

## 📜 License

This dataset and accompanying scripts are licensed under:\
**Creative Commons Attribution-NonCommercial 4.0 (CC-BY-NC 4.0)**\
Non-commercial use only, attribution required.\
[https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)

---

## 📖 Citation

```
Chopra, Muskaan, et al. "SynCED-EnDe: A Gold–Silver Dataset for Critical Error Detection in English→German MT." 2025.
Hugging Face Datasets: https://huggingface.co/datasets/moon712/SynCED_EnDe_2025
```

