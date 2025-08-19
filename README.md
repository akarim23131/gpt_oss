# GPT-OSS Semantic Analysis Project

## Overview

This project analyzes semantic relationships between query prompts and AI-generated responses using the GPT-OSS:20b model. It generates large-scale inference data, creates embeddings, computes cosine distances, and visualizes semantic relationships through interactive plots and tables.

## Workflow

### 1. Inference Generation
- Generates up to 1 million AI inferences using 174 diverse prompts (emotions, objects, actions, concepts)
- Uses local Ollama GPT-OSS:20b model with retry logic and error handling
- Saves results to organized text files (1000 inferences per file)

### 2. Embedding Creation
- Extracts query embeddings using OpenAI's text-embedding-3-small model
- Chunks long outputs into sentences (400 words, 100 word overlap)
- Creates embeddings for each sentence chunk
- Stores embeddings and text chunks in structured format

### 3. Semantic Distance Analysis
- Computes cosine distances between query embeddings and corresponding sentence embeddings
- Generates CSV files with sentence-level distance measurements
- Covers first 10 inferences for detailed analysis

### 4. Automated Labeling
- Uses GPT-OSS:20b to automatically classify each sentence with 2-3 word domain labels
- Applies intelligent prompting for consistent categorization
- Saves labeled output for visualization

### 5. Visualization & Analysis
- Creates publication-ready scatter plots showing semantic distance distributions
- Generates interactive tables with sentence labels and distances
- Provides box plots for statistical summary of distance ranges

## Key Components

- **`gpt_oss_test.ipynb`**: Main Jupyter notebook containing the entire pipeline
- **`embeddings/`**: Generated embeddings, chunks, distances, and labels
- **`inferencee/`**: Raw inference data from GPT-OSS model
- **`requirements.txt`**: Python dependencies

## Dependencies

- OpenAI API (for embeddings)
- Ollama (for GPT-OSS:20b model)
- NumPy, SciPy, Matplotlib, Pandas

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Ollama with GPT-OSS:20b model:
   ```bash
   ollama pull gpt-oss:20b
   ```

3. Configure OpenAI API key for embeddings

## Usage

Run the main Jupyter notebook to execute the complete pipeline:

```bash
jupyter notebook gpt_oss_test.ipynb
```

The notebook will guide you through each step of the semantic analysis process.