# CodeBART: Neural Code Summarization with PLBART and CodeT5

Fine-tuning and comparing two pre-trained code-aware encoder-decoder transformers, **PLBART** and
**CodeT5**, on the Python subset of the **CodeXGLUE code-to-text** benchmark. Given the body of a
Python function, the models generate a one-sentence natural-language summary (the first line of its
docstring).

The whole project lives in a single, self-contained notebook, [`codebart.ipynb`](codebart.ipynb),
designed to run end to end in one Kaggle GPU session.

## Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Method](#method)
- [Avoiding data leakage](#avoiding-data-leakage)
- [Results](#results)
- [Running the notebook](#running-the-notebook)
- [Using a fine-tuned model](#using-a-fine-tuned-model)
- [Repository layout](#repository-layout)
- [References](#references)

## Overview

Automatic code summarization helps with documentation, code review and comprehension of unfamiliar
code bases. This project treats it as sequence-to-sequence generation and asks a practical question:
starting from publicly available pre-trained code models and a modest amount of fine-tuning data, how
well do PLBART and CodeT5 summarize Python functions, and how do they compare under identical
conditions?

Both models go through exactly the same data preparation, training loop and evaluation code, so the
comparison is fair. Particular care is taken to keep the evaluation honest; see
[Avoiding data leakage](#avoiding-data-leakage).

## Dataset

[CodeXGLUE code-to-text](https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text)
(Python configuration), derived from CodeSearchNet. The full split sizes are:

| split | functions |
|---|---|
| train | 251,820 |
| validation | 13,914 |
| test | 14,918 |

Following the benchmark's official recipe:

- **input**: `code_tokens`, the function tokenized with its docstring and comments removed, joined with spaces;
- **target**: `docstring_tokens`, the tokenized first paragraph of the docstring, joined with spaces.

The notebook fine-tunes on a seeded random subset of each split (10% by default, configurable via
`SUBSET_FRACTION`) so that the full experiment fits into a single GPU session. Splits are shuffled
before subsampling because the dataset is ordered by repository.

## Method

| | PLBART | CodeT5 |
|---|---|---|
| checkpoint | `uclanlp/plbart-base` | `Salesforce/codet5-base` |
| architecture | BART-style denoising seq2seq, pre-trained on Java, Python and English | T5 pre-trained on code with identifier-aware objectives |
| parameters | ~140M | ~220M |
| batch size | 16 | 8 |

Shared setup:

- max input length 384 tokens, max summary length 128 tokens, dynamic padding per batch;
- AdamW (lr 3e-5, weight decay 0.01), linear warm-up over 6% of steps then linear decay, gradient clipping at 1.0;
- up to 10 epochs; after every epoch the validation set is decoded and the checkpoint with the best validation BLEU is kept (the CodeXGLUE selection criterion), with early stopping after 2 epochs without improvement;
- beam search with 4 beams at inference time;
- metrics: corpus BLEU-4, mean sentence-level BLEU-4 with add-one smoothing on higher-order n-grams (the scheme of the CodeXGLUE evaluator) and ROUGE-1/2/L F-measure, all on a 0-100 scale.

Both checkpoints are **pre-trained only**. Hugging Face also hosts task-fine-tuned variants for this
benchmark (for example `uclanlp/plbart-python-en_XX` and `Salesforce/codet5-base-multi-sum`); they
are deliberately not used, since fine-tuning them again would measure nothing.

## Avoiding data leakage

Code summarization is easy to get wrong in ways that inflate scores silently rather than failing
loudly. The notebook addresses each of the following explicitly:

| concern | decision |
|---|---|
| Target visible in the input | The raw `code` field contains the docstring. The input is `code_tokens` instead, and the notebook *measures* how often the target appears verbatim in the input for both fields before training. |
| Task-fine-tuned starting checkpoints | Only pre-trained checkpoints are used. |
| Unrepresentative subset | Splits are shuffled with a fixed seed before subsampling. |
| Loss on padding | Labels are padded with `-100`, which the loss ignores. |
| Train/inference mismatch (PLBART) | Targets are tokenized in target mode so they end with the `__en_XX__` language code, the same token the decoder is started with. |
| Overfitting | Model selection and early stopping on validation BLEU; the best epoch is evaluated, not the last. |
| Test-set reuse | Model selection uses the validation set only; the test set is decoded and scored exactly once per model. |
| Metric bookkeeping | A fresh scorer and aggregator per evaluation; references are the untruncated dataset strings. |
| Reproducibility | Global seed, seeded shuffling and data-loader order, all hyper-parameters in one configuration cell. |

## Results

Scores are produced by the notebook (Section 6) and written to `checkpoints/results.csv`. Fill in
this table from your run; the numbers depend on `SUBSET_FRACTION` and the GPU session.

| model | split | BLEU (corpus) | BLEU (sentence avg) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|
| PLBART | test | | | | | |
| CodeT5 | test | | | | | |

For context, the published CodeXGLUE code-to-text leaderboard reports smoothed BLEU-4 of roughly
19-20 on Python for PLBART and CodeT5 fine-tuned on the **full** training split. Scores from a 10%
subset are expected to be lower; the notebook's training curves and side-by-side prediction table
show where each model succeeds and fails.

## Running the notebook

The notebook was designed for Kaggle and tested on its current image (Python 3.13, PyTorch 2.11,
transformers 5.20, datasets 4.8); the pre-installed packages are used as they are. The code also runs
on transformers 4.4x.

1. Create a new Kaggle notebook and upload `codebart.ipynb` (*File → Import Notebook*).
2. In *Settings*, set **Accelerator** to *GPU T4 x2* or *GPU P100* and turn **Internet** *On*
   (the dataset and model weights come from the Hugging Face Hub).
3. *Run All*. With the default 10% subset the whole notebook takes roughly 1.5-2.5 hours on a T4;
   `SUBSET_FRACTION = 0.01` gives a ~20 minute smoke run.
4. Fine-tuned checkpoints, `results.csv` and `test_predictions.csv` are written to
   `/kaggle/working/checkpoints` and appear in the notebook's *Output* tab.

Everything Kaggle-specific is marked with a `# Kaggle:` comment inside the notebook. To run
elsewhere, any machine with a 16 GB GPU and the following packages will do:

```bash
pip install torch transformers datasets sentencepiece rouge-score nltk pandas matplotlib
```

Change `SUBSET_FRACTION` in the configuration cell to trade training time for model quality;
runtime scales roughly linearly with it.

## Using a fine-tuned model

The saved checkpoints are standard Hugging Face models. Section 7 of the notebook defines a
`summarize()` helper that applies the same preprocessing as training (docstring removal with `ast`,
tokenization with Python's own tokenizer) to arbitrary source code:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("checkpoints/codet5")
model = AutoModelForSeq2SeqLM.from_pretrained("checkpoints/codet5")

print(summarize(source_code, model, tokenizer))   # summarize() is defined in the notebook
```

For the PLBART checkpoint, pass `decoder_start_token_id=tokenizer.convert_tokens_to_ids("__en_XX__")`
as an extra argument so the decoder is started with the English language code.

## Repository layout

```
codebart.ipynb   the complete experiment: data, leakage check, training, evaluation, inference
readme.md        this file
LICENSE          MIT
```

## References

- Lu et al., *CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation*, 2021.
- Ahmad et al., *Unified Pre-training for Program Understanding and Generation* (PLBART), NAACL 2021.
- Wang et al., *CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation*, EMNLP 2021.
- Husain et al., *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search*, 2019.

## License

MIT, see [LICENSE](LICENSE).
