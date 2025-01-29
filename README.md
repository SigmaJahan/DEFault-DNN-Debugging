# **Replication Package for DEFault**

Welcome to the replication package for **DEFault**, a framework designed to improve the detection and diagnosis of faults in Deep Neural Networks (DNNs). This repository provides all the necessary code and data to reproduce the experiments from our paper, which has been accepted in ICSE - Research Track 2025.

**"Improved Detection and Diagnosis of Faults in Deep Neural Networks using Hierarchical and Explainable Classification."**

Pre-print of the paper can be found: https://arxiv.org/abs/2501.12560

---
## **How DEFault Works**

DEFault is a hierarchical classification framework that improves the detection and diagnosis of faults in Deep Neural Networks (DNNs) by leveraging both static and dynamic analysis. The workflow of DEFault consists of three primary stages:

### **1. Fault Detection**  
- DEFault monitors runtime features such as loss trends, activation statistics, and gradient behaviors during model training.  
- It detects whether a given DNN program contains one or more faults using a trained classifier.

### **2. Fault Categorization**  
- Once a fault is detected, DEFault categorizes it into one or more fault types across seven categories:  
  - Hyperparameter  
  - Loss  
  - Activation  
  - Layer  
  - Optimizer  
  - Weight  
  - Regularization  
- This step is performed using multiple binary classifiers trained on extracted static and dynamic features.

### **3. Root Cause Analysis**  
- For deeper fault diagnosis, DEFault employs an explainer module leveraging SHAP to pinpoint the most influential static and dynamic features contributing to the fault.  
- This helps developers understand the exact source of issues, such as incorrect hyperparameters or misconfigured layers.

The figure below illustrates the workflow of DEFault, showing its fault detection, categorization, and root cause analysis processes.

<img width="1434" alt="schematicupdated" src="https://github.com/user-attachments/assets/fba74550-da1d-40b2-b0ab-bb1dddd1d4da" />

---

## **Table of Contents**
1. [Contents](#contents)
2. [Requirements](#requirements)
3. [Usage: Whole Dataset vs. Sample Data](#usage-whole-dataset-vs-sample-data)
    - [1. Data Collection](#1-data-collection)
    - [2. Feature Extraction](#2-feature-extraction)
    - [3. Model Training](#3-model-training)
    - [4. Model Evaluation](#4-model-evaluation)
    - [5. Case Studies](#5-case-studies)
4. [Usage: Sample Data](#usage-sample-data)
5. [Quick Start Guide](#quick-start-guide)
6. [How to Cite](#how-to-cite)
7. [Authors](#authors)
8. [Licensing Information](#licensing-information)

---

## **Contents**

### Repository Structure

- **`0_Artifact_Testing/`**: Scripts to run the experiment on one sample DNN model.
- **`a_Data_Collection/`**: Scripts to collect and process StackOverflow posts for creating the dataset.
- **`b_Fault_Seeding/`**: Scripts to inject faults into DNN programs using the extended DeepCrime framework:
  - **Part 1-DC**: Original DeepCrime code.
  - **Part 2-EFI**: Extended framework for more fault types, including convolutional and recurrent models.
- **`c_Feature_Extraction/`**: Scripts for extracting features from DNN programs:
  - **Dynamic**: Runtime feature extraction.
  - **Static**: Structural feature extraction.
- **`d_DEFault/`**: Implementation of DEFault:
  - **A_Detection/**: Fault detection scripts.
  - **B_Categorization/**: Fault categorization scripts.
  - **C_RootCauseAnalysis/**: Root cause analysis scripts.
- **`e_Evaluation/`**: Scripts to evaluate DEFault on real-world and seeded faults.
- **`f_Figures/`**: Figures used in the paper.
- **`g_Dataset/`**: Labeled datasets for training and testing.
- **`h_CohenKappaAnalysis/`**: Scripts for dataset consistency validation using Cohen's Kappa.
- **`i_CaseStudy/`**: Scripts for case studies on real-world models (e.g., PixelCNN).
- **`j_HPC_Slurm/`**: Example Script for Slurm job on Compute Canada with all the configuration.

---

## **Requirements**

### **Operating System**
Tested on:
- Ubuntu 20.04 LTS or later
- HPC environments such as Compute Canada (Graham Cluster)

Compatible with:
- Windows 10/11 (via Windows Subsystem for Linux - WSL2)
- macOS Monterey (M1/M2 support may require additional configuration)

### **Hardware Requirements**

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 10 GB

**Recommended:**
- GPU: NVIDIA with CUDA support
- HPC access (e.g., Compute Canada) for the complete experiment

### **Software Requirements**

- **Python Version:** 3.10 or later  
- **Dependencies:** Install via `requirements.txt`:  
  ```bash
  pip install -r requirements.txt
  ```

**Recommended:** Create a virtual environment before installation:  
```bash
python -m venv default_env
source default_env/bin/activate   # On macOS/Linux
default_env\Scripts\activate      # On Windows
```

---

## **Usage: Complete Experiment vs. Lightweight Verification**

**Important:**  
- Running the **Complete Experiment** on the whole dataset requires significant computational resources and time.
- Running the **Lightweight Verification** on a sample DNN program is **recommended**, as it provides a quick and effective way to verify the framework's functionality.

---

## **Usage: Lightweight Verification**

The 0_Artifact_Testing directory provides all necessary artifacts to reproduce case study results with minimal computational overhead. It includes:

- **Data**
- **Pre-trained models**
- **Evaluation scripts**
- **Configuration files**
- **Requirements**
- **README instructions**

The expected result for the sample data is provided inside the directory.

---

## **Usage: Complete Experiment**

### **1. Data Collection**

1. **Download Original DNN Programs**:  
   - Download the 60 original DNN programs that are collected from StackOverflow:  
     [Download Link](https://bit.ly/3Cw0vOB).
2. **Inject Faults**:  
   ```bash
   cd b_Fault_Seeding
   python Fault_Seeding_Script.py
   ```

### **2. Feature Extraction**

```bash
cd c_Feature_Extraction/Static
python Static_Feature_Extraction.py
```

```bash
cd c_Feature_Extraction/Dynamic
python Dynamic_Feature_Extraction.py
```

### **3. Model Training**

```bash
cd d_DEFault/A_Detection
python Fault_Detection.py
```

### **4. Model Evaluation**

- Download the evaluation benchmark (Collected from [DeepFD](https://github.com/ArabelaTso/DeepFD/)
  [Download Link](https://bit.ly/3CQPozK)).

```bash
cd e_Evaluation
python Fault_Evaluation_Detection_Diagnosis.py
```

### **5. Case Studies**

```bash
cd i_CaseStudy
python Feature_Extraction_CaseStudy.py
python PixelCNN_Analysis.py
```

---

## **How to Cite**

If you use DEFault in your research, please cite our paper:

```
@inproceedings{default2025,
  author    = {Sigma Jahan and Mehil B Shah and Parvez Mahbub and Mohammad Masudur Rahman},
  title     = {Improved Detection and Diagnosis of Faults in Deep Neural Networks using Hierarchical and Explainable Classification},
  booktitle = {Proceedings of the International Conference on Software Engineering (ICSE)},
  year      = {2025},
  pages     = {13},
  publisher = {IEEE}
}
```

---

## **Authors**

- **Sigma Jahan** - Dalhousie University, sigma.jahan@dal.ca  
- **Mehil B Shah** - Dalhousie University, shahmehil@dal.ca  
- **Parvez Mahbub** - Dalhousie University, parvezmrobin@dal.ca  
- **Mohammad Masudur Rahman** - Dalhousie University, masud.rahman@dal.ca  

---

## **Licensing Information**

This project is licensed under the **MIT License**, allowing free usage, modification, and distribution. We encourage collaboration and knowledge sharing. See `LICENSE` for more details.
