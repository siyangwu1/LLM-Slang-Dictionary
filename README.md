# How do Language Models Generate Slang: A Systematic Comparison Between Human and Machine-Generated Slang Usages
## Authors

- **Siyang Wu**  
  Data Science Institute, University of Chicago, Chicago, Illinois  
  [siyangwu@uchicago.edu](mailto:siyangwu@uchicago.edu)

- **Zhewei Sun**  
  Toyota Technological Institute at Chicago, Chicago, Illinois  
  [zsun@ttic.edu](mailto:zsun@ttic.edu)

## Overview

This repository accompanies the paper:  
**"How do Language Models Generate Slang: A Systematic Comparison Between Human and Machine-Generated Slang Usages."**

All authorship information has been anonymized for double-blind peer review.

This repository contains:

- Code for generating, evaluating, and analyzing slang terms from LLMs
- Cleaned and structured datasets of both human-written and LLM-generated slang
- Metrics and tools to quantify semantic novelty and morphological coherence

---

## Repository Structure

- `code/`
  - `generation.py` – Controlled and uncontrolled slang generation pipeline
  - `novelty.py` – Semantic novelty scoring script
  - `coinage_coherence.py` – Compound word coherence analysis

- `data/`

  - `gpt4o_CF.csv`, `llama_8b-it_UF.csv`, etc. – LLM-generated slang under various generation schemas



---

## Code Descriptions (`code/`)

### `generation.py`

Generates slang entries using prompts sent to language models like **GPT-4o** and **LLaMA-8B**. Supports three modes:
- **Freeform**: Open-ended slang invention
- **Reuse**: Repurposing existing words
- **Coinage**: Inventing novel lexical forms

We used OSD sense clusters to guide definitions in controlled modes. Outputs are saved with metadata including source model, definition used, and cluster ID.

---

### `novelty.py`

Quantifies **semantic novelty** by computing the mean **Euclidean distance** between a slang word’s definition and standard dictionary definitions of the same word. It:
- Uses SBERT for embedding comparisons
- Distinguishes between human and machine sources
- Outputs per-source novelty scores for comparison

---

### `coinage_coherence.py`

Analyzes the **morphological coherence** of compound slang coinages. It:
1. Trains Morfessor on slang vocabulary
2. Classifies words as compounds
3. Computes average distance between the slang definition and constituent morphemes' dictionary senses
4. Summarizes coherence by generation source

---

## Dataset Description (`data/`)

We provide **LLM-generated slang datasets**. Datasets are labeled by:
- **Model Identifier** (e.g., `gpt4o`, `llama_8b-it`)
- **Schema Code**:
  - First letter:
    - `C`: Controlled (guided by human definition)
    - `U`: Uncontrolled (freeform generation)
  - Second letter:
    - `F`: Freeform
    - `C`: Coinage
    - `R`: Reuse

Each `.csv` file contains slang entries with associated definitions and usage examples.

---

## Valid Entry Summary by File

| File Name                      | Valid Rows |
|-------------------------------|------------|
| `online_slang_dictionary.csv` | 9115       |
| `llama_8b-it_UC.csv`          | 1009       |
| `llama_8b-it_CF.csv`          | 5855       |
| `gpt4o_UR.csv`                | 1011       |
| `gpt4o_CC.csv`                | 9082       |
| `llama_8b-it_CR.csv`          | 2092       |
| `gpt4o_UF.csv`                | 1000       |
| `llama_8b-it_CC.csv`          | 8062       |
| `llama_8b-it_UF.csv`          | 1004       |
| `gpt4o_CR.csv`                | 8853       |
| `gpt4o_UC.csv`                | 1000       |
| `llama_8b-it_UR.csv`          | 1000       |
| `gpt4o_CF.csv`                | 9114       |
| **Total**                     | **58,197** |

---

## Format and Fields

All `.csv` files are encoded in UTF-8 and share the following fields:
- `word`: The slang term
- `definition`: The definition of the slang
- `usage_context`: One or more usage examples

Invalid rows have been removed and columns normalized for consistency.

---

## Usage Guidelines

You may use this repository to:

- Evaluate LLM creativity in slang generation
- Compare model generations to human-written slang
- Explore linguistic strategies like reuse, coinage, and freeform creation
- Run controlled prompting experiments using OSD reference clusters

---

## License and Citation

This repository is released for **academic research use only**.

Please cite the paper if you use the code or dataset. Citation details will be provided upon publication.
