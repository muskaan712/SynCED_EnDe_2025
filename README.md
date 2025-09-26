# SynCED-EnDe — Pipeline, Raw Data & Scripts (GitHub Companion)

This repository provides **pipeline scripts and raw/intermediate files** for building the
[SynCED-EnDe dataset](https://huggingface.co/datasets/your-username/synced-ende).

The **clean, benchmark-ready release** (train silver + eval gold) is on Hugging Face.  
This repo is for **reproducibility and transparency**.

---

## 📂 Repository Structure

```
scripts/                 # Core dataset creation pipeline
  01_labels.py              # defines binary + multi-class labels
  02_inject.py              # performs error injection
  03_reinject.py            # second-pass injection/fixes
  04_skim.py                # cleaning & filtering
  05_block_evalrows.py      # prevents eval leakage
  06_final.py               # assembles final TSVs

extras/                  # Helper & debug scripts
  data_check.py             # sanity checks (counts, distributions)
  data_scrape.py            # (if shareable) source scraping
  train_data/               # legacy scripts
    check.py
    master.py

data/                    # Raw & intermediate data files
  train-silver/raw/...
  eval-gold/raw/...

docs/
  annotation_protocol.pdf   # annotation guidelines
  labels.json               # schema for labels
```

---

## 🚀 Pipeline Usage

Run the scripts in order:

1. **Define labels**
   ```bash
   python scripts/01_labels.py
   ```

2. **Inject errors**
   ```bash
   python scripts/02_inject.py
   ```

3. **Re-inject (optional second pass)**
   ```bash
   python scripts/03_reinject.py
   ```

4. **Clean dataset**
   ```bash
   python scripts/04_skim.py
   ```

5. **Block evaluation rows**
   ```bash
   python scripts/05_block_evalrows.py
   ```

6. **Assemble final dataset**
   ```bash
   python scripts/06_final.py
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

⚠️ **Note:** These files may contain earlier/unverified labels.  
They are **not for benchmarking** — use only for reproducing pipeline steps.

---

## 📘 Documentation
- `docs/annotation_protocol.pdf` → annotation guidelines  
- `docs/labels.json` → schema for binary, multi-class, and 5-dimension ratings  

---

## 📜 License
This dataset and accompanying scripts are licensed under:  
**Creative Commons Attribution-NonCommercial 4.0 (CC-BY-NC 4.0)**  
Non-commercial use only, attribution required.  
<https://creativecommons.org/licenses/by-nc/4.0/>

---

## 📖 Citation
```
Chopra, Muskaan, et al. "SynCED-EnDe: A Gold–Silver Dataset for Critical Error Detection in English→German MT." 2025.
Hugging Face Datasets: https://huggingface.co/datasets/your-username/synced-ende
