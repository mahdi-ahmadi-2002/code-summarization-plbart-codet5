# CodeBART: An Efficient BART-based Model for Code Summarization

## 📜 Abstract
This project, **CodeBART**, presents a fine-tuned encoder-decoder model for the task of automatic code summarization. The goal is to generate concise, human-readable docstrings (summaries) from Python source code. Leveraging the power of the pre-trained **BART-base** model from Hugging Face Transformers, this work demonstrates an effective and computationally efficient approach to sequence-to-sequence modeling in the software engineering domain. The model was fine-tuned and evaluated on the **CodeSearchNet** dataset, achieving strong performance on established metrics like BLEU and ROUGE.

---

## 📋 Table of Contents
- [Introduction](#-introduction)
- [Dataset and Preprocessing](#-dataset-and-preprocessing)
- [Methodology](#-methodology)
  - [Model Architecture: BART](#-model-architecture-bart)
  - [Fine-Tuning Process](#-fine-tuning-process)
- [Results](#-results)
  - [Quantitative Metrics](#-quantitative-metrics)
  - [Example Generation](#-example-generation)
- [How to Use](#-how-to-use)
- [Conclusion](#-conclusion)

---

## 🏛️ Introduction
Automatic code summarization is a critical task in software engineering, aiding in code documentation, comprehension, and maintenance. While large language models have shown great promise, they often require extensive computational resources. This project explores an alternative by fine-tuning a more compact, pre-trained model, **BART (Bidirectional and Auto-Regressive Transformers)**, specifically for generating Python docstrings. The resulting model, CodeBART, serves as an efficient yet powerful tool for this natural language generation task.

---

## 📊 Dataset and Preprocessing

- **Dataset**: The model was trained on the Python subset of the **CodeSearchNet** dataset. This large-scale dataset is a standard benchmark for code-related tasks and contains a rich collection of function-docstring pairs.
  -   **Input**: The full Python function source code (`whole_func_string`).
  -   **Target**: The corresponding natural language docstring (`func_documentation_string`).

- **Preprocessing**: The `transformers` and `datasets` libraries from Hugging Face were used for all data handling.
    1.  **Tokenization**: A `RobertaTokenizer` was used to tokenize both the source code and the natural language summaries.
    2.  **Formatting**: The tokenized code-summary pairs were formatted into the input/output structure required by the sequence-to-sequence model.

---

## ⚙️ Methodology

### Model Architecture: BART
CodeBART is not a new architecture but a fine-tuned version of the **`facebook/bart-base`** model. The BART architecture is uniquely suited for summarization tasks due to its design as a denoising autoencoder.
-   **Bidirectional Encoder**: Processes the input source code, allowing it to build a rich, context-aware representation by looking at the code in both forward and backward directions.
-   **Auto-Regressive Decoder**: Generates the natural language summary token-by-token, conditioning each new token on the encoder's representation of the code and the tokens it has previously generated.

This structure allows the model to learn a deep understanding of the source code's syntax and semantics and translate it into a coherent summary.

### Fine-Tuning Process
The pre-trained BART model was fine-tuned using the high-level `Seq2SeqTrainer` API from the `transformers` library.
-   **Optimizer**: `AdamW`
-   **Learning Rate**: $2 \times 10^{-5}$ with weight decay.
-   **Batch Size**: `16` (8 per device).
-   **Epochs**: The model was trained for **3 epochs**.
-   **Evaluation Metrics**: Performance was evaluated using standard text-generation metrics: **ROUGE** and **BLEU**.

---

## 📈 Results

### Quantitative Metrics
The model's performance was evaluated on the test set of CodeSearchNet, achieving the following scores (as detailed in the accompanying report):

| Metric      | Score      |
| ----------- | ---------- |
| ROUGE-1     | 43.21      |
| ROUGE-2     | 21.05      |
| ROUGE-L     | 38.97      |
| METEOR      | 22.80      |
| BLEU        | 16.51      |

### Example Generation
The following is an example of the model's summarization capability on a sample Python function:

**Input Code:**
```python
def check_file_balance(filename):
    """Checks if a file is balanced.
    A file is balanced if for every open parenthesis, there is a closing parenthesis.
    This function reads the file and returns True if the file is balanced, False otherwise.
    """
    with open(filename, 'r') as f:
        stack = []
        for line in f:
            for char in line:
                if char == '(':
                    stack.append(char)
                elif char == ')':
                    if not stack:
                        return False
                    stack.pop()
    return not stack
```

**Generated Summary:**
```
Checks if a file is balanced.
```

---

## 🚀 How to Use

1.  **Install Dependencies:**
    ```bash
    pip install torch transformers datasets evaluate sacrebleu
    ```
2.  **Run Inference:**
    Load the fine-tuned model and tokenizer, then use a pipeline or a custom generation function to summarize new code snippets.
    ```python
    from transformers import pipeline

    # Load the fine-tuned model (replace with your model path/hub name)
    summarizer = pipeline("summarization", model="your-finetuned-model")

    code_snippet = "def example_function(n):\\n    return n * 2"
    summary = summarizer(code_snippet, max_length=50, min_length=5, do_sample=False)
    print(summary[0]['summary_text'])
    ```

---

## 💡 Conclusion
This project successfully demonstrates that fine-tuning a medium-sized, pre-trained transformer model like BART-base offers a powerful and resource-efficient solution for the complex task of code summarization. CodeBART effectively learns to map code semantics to natural language, providing a valuable tool for automated software documentation.