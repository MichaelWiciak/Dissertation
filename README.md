# Code Completion with Transformers

A dive into what happens when you fine-tune pre-trained language models (RoBERTa, CodeBERT, UniXcoder) for code completion using Masked Language Modeling, and whether enriching the training data with comments or AST structure actually helps.

Click [here](https://drive.google.com/file/d/1y1hG72kHNpezTbFJcssOjT479SAj-Rs9/view?usp=sharing) for the three trained models.

Check out the [Dissertation.pdf](./Dissertation.pdf) which contains the results and analysis in more detail, including the literature review, methodology, and discussion of findings.

## What This Project Explores

Given a code snippet with a token replaced by `<mask>`, can we train a model to predict what belongs there?

More specifically:

- Does fine-tuning a general language model (RoBERTa) on code make it better?
- Does adding docstrings/comments to the training data help?
- Does incorporating the Abstract Syntax Tree (AST) structure help?
- How do these compare to models pre-trained specifically on code (CodeBERT)?
- How do the models perform on a programming language it wasn't trained on?

## The Interesting Bits

### AST Integration

The AST variant (CCA) performed best among fine-tuned models, suggesting structural information does help. The approach parsed code into AST using Python's built-in `ast` module and JavaScript's `esprima`, then fed the AST representation alongside the code tokens.

### Multi-Language Evaluation

Tested across Python, JavaScript, and C++ (using IBM's CodeNet dataset) to see if findings held across languages. The AST-based approach showed consistent improvements across languages.

### Timing Analysis

Also measured inference time vs token length - useful context for understanding practical trade-offs alongside accuracy.

## How It Works

### Masked Language Modeling for Code Completion

The task is straightforward: take code, mask one token, train the model to predict what was masked. This is the same objective as BERT's pre-training, applied to code-specific fine-tuning.

```
Input:  def add(x, y): return x <mask> y
Output: +
```

### Three Dataset Variants

1. **Code Only** - Raw code tokens with comments/docstrings stripped
2. **Code + Comments** - Code with docstrings included in the input
3. **Comments + AST** - Comments + AST structure representation of the code

### LoRA Fine-Tuning

Used Low-Rank Adaptation (LoRA) for efficient fine-tuning, updating a subset of parameters rather than full model fine-tuning.

### Datasets

- **CodeSearchNet** - Python and JavaScript functions from GitHub
- **CodeNet** - C++ solutions from IBM's Project CodeNet

## Project Structure

```
TransformerCodeCompletion/
├── Models/                    # Training scripts and model configs
│   ├── main.py                # Code-only training with LoRA
│   ├── CodeOnlyDataset.py     # PyTorch dataset for code-only
│   ├── CodeCommentsDataset.py # Dataset for code + comments
│   ├── CodeCommentsASTDataset.py  # Dataset with AST structure
│   ├── loraSetup.py           # LoRA configuration
│   └── listOfExamples.txt     # Example completions
│
├── DatasetManipulations/      # Data preprocessing
│   ├── preprocessing.py       # AST validation, masking, dataset creation
│   └── subsetGeneration.py   # Generate dataset subsets
│
├── Comparing_Models/           # Evaluation framework
│   ├── Comparison.py          # Main evaluation script
│   ├── results.json           # Aggregated comparison results
│   └── notebook.ipynb         # Visualizations and analysis
│
├── Testing/                   # Basic sanity tests
└── Dissertation.pdf           # Full write-up
```

---

## Getting Started

### Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Preparing Data

The project works with CodeSearchNet and CodeNet datasets. After obtaining the data:

```bash
# Preprocess a dataset (validates AST, creates mask, generates variants)
python DatasetManipulations/preprocessing.py /path/to/your/data.jsonl

# Generate training/validation/test splits
python DatasetManipulations/subsetGeneration.py 80 train
```

### Training a Model

```bash
# Fine-tune on code only
python Models/main.py
```

### Running Evaluation

```bash
# Compare all models
python Comparing_Models/Comparison.py
```

---

## Visualizations

The `Comparing_Models/notebook.ipynb` contains:

- Accuracy comparison bar charts across all models
- Token length vs inference time line plots
- Masking length experiments
- Per-language breakdowns

## More Detail

For the full story, including:

- Detailed literature review
- Methodology and experimental design
- Extended results and error analysis
- Discussion of findings and limitations

Check out the [Dissertation.pdf](./Dissertation.pdf).
