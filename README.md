# GATESynergy: integrating molecular global-local aggregator and hierarchical gene-gated encoder for drug synergy prediction
## 1. Overview
The code for paper "GATESynergy: integrating molecular global-local aggregator and hierarchical gene-gated encoder for drug synergy prediction". The repository is organized as follows:
- data/ contains the datasets used in the paper;
- code/train.py contains training and testing code;
- code/process_data.py contains the preprocess of data;
- code/dataset.py contains the dataset construction and preprocessing for drug graph data.
- code/model.py contains GATESynergy's model layer;

## 2. Dependencies
- Python 3.9.12
- PyTorch 2.1.0 + CUDA 12.1
- torch-geometric 2.5.1
- NumPy 1.24.1
- Pandas 2.2.1
- SciPy 1.12.0
- scikit-learn 0.24.2
- RDKit 2023.9.3

## 3. Quick Start
Here we provide to predict drug synergy:
1. Download and upzip our data and code files
2. Run "train.py"

## 4. Contacts
If you have any questions, please email Ding Jiana (dingjn24@mails.jlu.edu.cn)

## 5. Workflow of GATESynergy
![Figure2](https://github.com/user-attachments/assets/3e21878e-d06d-4344-a155-aac1cdd354bb)
