# Generative AI Project — Psalm- and Proverb-Style Text with a Character-Level GPT

## Project Description

A generative AI system built in PyTorch in a single Jupyter notebook, `generative_model.ipynb`.
A small decoder-only Transformer (a GPT-style language model with 10.8 million parameters) is trained from scratch, character by character, to generate new Spanish verses in the style of the Psalms and Proverbs of the 1909 Reina-Valera Bible.
Training has two phases — pre-training on the whole Bible to learn 1909 Spanish, then fine-tuning on Psalms and Proverbs with early stopping to learn their style — and the book code at the start of each line (`PSA:` or `PRO:`) works as a style prompt.
The notebook generates many samples, compares temperatures and training stages, and measures invented words, verbatim memorization of the training text and repetition, with a qualitative evaluation of strengths and failure cases.

**Dataset:** *Santa Biblia — Reina Valera 1909*, public domain, verse-per-line plain-text edition from eBible.org — https://ebible.org/find/details.php?id=spaRV1909 (download: https://ebible.org/Scriptures/spaRV1909_vpl.zip).
The file used, `data/spaRV1909_vpl.txt` (31,102 verses, 4.2 MB), and the publisher's description, `data/spaRV1909_about.htm`, are included in this repository.

## How to Run the Project

Requirements: **Python 3.12 or newer** and Git. The pinned versions in `requirements.txt` (for example `numpy==2.5.3` and `contourpy==1.4.0`) do not install on Python 3.11 or older.

```bash
git clone https://github.com/arturoappp/generative-ai-applications-project.git
cd generative-ai-applications-project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook generative_model.ipynb
```

Then run all cells (Kernel → Restart & Run All). The notebook runs top to bottom without errors and writes three figures to `figures/`.

**GPU (strongly recommended).** `requirements.txt` pins `torch==2.11.0`, which installs a CUDA build on Linux and a CPU build on Windows from PyPI. The results in the report were produced on an NVIDIA RTX 3060 Ti, where the whole notebook (training plus about 1,000 generated verses for evaluation) takes about 30 minutes. To use an NVIDIA GPU on Windows, install the CUDA build first:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

On a CPU the notebook runs, but training is very slow: an independent CPU-only test measured about 7.5 seconds per training step, so the 4,600 steps take roughly 10 hours. Every random generator is seeded and PyTorch's deterministic algorithms are enabled, so the generated samples are reproducible on the same hardware; exact numbers can differ slightly between CPU and GPU or between GPU models.

To regenerate the dependency file from the project environment:

```bash
pip freeze > requirements.txt
```

## Repository Structure

```
generative_model.ipynb                  # data inspection, preprocessing, GPT model, two-phase training, generation, evaluation, summary
Generative_AI_Analysis_Report.pdf       # written report with in-text citations and references
requirements.txt                        # exact package versions (pip freeze)
data/                                   # spaRV1909_vpl.txt (the corpus) and spaRV1909_about.htm (source and license note)
figures/                                # fig1_training_loss, fig2_temperature_diagnostics, fig3_model_comparison
```
