# VULDA
Official implementation of our paper: VULDA: Source Code Vulnerability Detection via Local Dependency Context Aggregation on Vulnerability-aware Code Mapping Graph
## Overview

This project implements **VULDA: Source Code Vulnerability Detection via Local Dependency Context Aggregation on Vulnerability-aware Code Mapping Graph**. It addresses the challenges of redundant information and insufficient cross-line semantic dependencies in large-scale source code vulnerability detection.

The method integrates two key components:

* **Vulnerability-aware Code Mapping Graph (VCMG)**: A graph that aligns multi-granularity semantics to line-level nodes, reducing redundancy and incorporating static heuristic rules and structural features to weight nodes. This enhances the model’s sensitivity to key vulnerability-related code.

* **Local Dependency Context Aggregation (LDCA)**: Aggregates both control-flow graph (CFG) and data-dependency graph (DDG) paths to capture dual-context information of logical and data semantics, improving the detection of complex vulnerability patterns.

---

## Scripts Description

| Script Name                      | Description                                                                                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `1_extract_functions.py`         | Extracts function-level code snippets from source files for analysis                                                                            |
| `2_parse_with_joern.py`          | Parses code with Joern to generate code property graphs (CPGs)                                                                                  |
| `3_normalize_code.py`            | Normalizes source code to a unified format to reduce noise                                                                                      |
| `4_generate_VCMG.py`             | Generates Vulnerability-aware Code Mapping Graph (VCMG), capturing multi-granularity semantics and weighting nodes based on heuristics          |
| `5_generate_LDCA_VCMG.py`        | Generates enhanced graphs by combining Local Dependency Context Aggregation (LDCA) with VCMG for richer vulnerability representation            |
| `6_data_processing.py`           | Processes and prepares datasets, converting graph and code data into model input formats                                                        |
| `7_train_and_evaluate_models.py` | Trains and evaluates various deep learning models (LSTM, BiLSTM, GRU, BiGRU, CNN) on the processed data, reporting multiple performance metrics |

---

## Usage

Run the scripts sequentially to process the source code and train vulnerability detection models:

```bash
python 1_extract_functions.py
python 2_parse_with_joern.py
python 3_normalize_code.py
python 4_generate_VCMG.py
python 5_generate_LDCA_VCMG.py
python 6_data_processing.py
python 7_train_and_evaluate_models.py
```

Adjust file paths and parameters inside each script as needed.

---

## Requirements

* Python 3.8
* PyTorch
* pandas
* scikit-learn
* Joern (for static code analysis)
* numpy

---

## Reference

This project is based on the method described in the paper:

**VULDA: Source Code Vulnerability Detection via Local Dependency Context Aggregation on Vulnerability-aware Code Mapping Graph**

Abstract:
*Vulnerability detection is crucial in software security. Existing methods suffer from redundancy and insufficient semantic dependencies, limiting detection performance. VULDA integrates a Vulnerability-aware Code Mapping Graph (VCMG) and Local Dependency Context Aggregation (LDCA) to address these issues. VCMG reduces graph redundancy and weights nodes via static heuristic rules, improving sensitivity to vulnerability code. LDCA aggregates control-flow and data-dependency paths, enhancing the model’s ability to express complex vulnerability patterns. Experimental results show VULDA outperforms baselines on datasets like SARD, Reveal, and FFmpeg+Qemu, with a 23.09% F1 score improvement on Reveal.*

---

