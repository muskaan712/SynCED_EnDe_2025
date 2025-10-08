# SynCED-EnDe — Pipeline, Raw Data & Scripts (GitHub Companion)

This repository provides **pipeline scripts and raw/intermediate files** for building the [SynCED-EnDe dataset](https://huggingface.co/datasets/moon712/SynCED_EnDe_2025).

The **clean, benchmark-ready release** (train silver + eval gold) is on Hugging Face.\
This repo is for **reproducibility and transparency**.

---

## 📂 Repository Structure

```
scripts/                    # Core dataset creation pipeline
  inject.py                 # inject controlled errors
  reinject.py               # retry injection for missing cases
  skim.py                   # preview dataset (sanity check)
  block_evalrows.py         # filter out scraped evaluation rows from train set 
  final.py                  # assemble final dataset TSVs
  data_check.py             # sanity checks (counts, distributions, label balance)
  data_scrape.py            # scrape GOV.UK + Stack Exchange

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

1. **Scrape Data**

   ```bash
   python scripts/data_scrape.py
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

6. **Judge & quantify translations (LLM-based)**
   ```bash
   python scripts/judge_quantify.py

7. **Assemble final dataset**

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
**Creative Commons Attribution- 4.0 (CC-BY 4.0)**\
Attribution required.

---

## 📖 Citation

```
@article{chopra2025syncedende,
  title={SynCED-EnDe 2025: A Synthetic and Curated English - German Dataset for Critical Error Detection in Machine Translation},
  author={Chopra, M. and others},
  journal={Hugging Face Papers},
  year={2025},
  url={https://huggingface.co/papers/2510.05144}
}

```

