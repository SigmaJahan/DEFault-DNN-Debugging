# **Replication Package for DEFault**

Welcome to the replication package for **DEFault**, a framework designed to improve the detection and diagnosis of faults in Deep Neural Networks (DNNs). This repository provides all the necessary code and data to reproduce the experiments from our paper, which has been accepted in ICSE - Research Track 2025. Pre-print: https://arxiv.org/abs/2501.12560

**"Improved Detection and Diagnosis of Faults in Deep Neural Networks using Hierarchical and Explainable Classification."**

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
    - [1. Data](#data)
    - [2. Models](#models)
    - [3. Evaluation Scripts](#evaluation-scripts)
    - [4. Config](#config)
    - [5. Requirements](#requirements)
    - [6. README.md](#readme)
5. [Quick Start Guide](#quick-start-guide)
6. [Licensing Information](#licensing-information)

---

## **Contents**

### Repository Structure

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
- **`g_Dataset/`**: Labeled datasets for training and evaluation.
- **`h_CohenKappaAnalysis/`**: Scripts for dataset consistency validation using Cohen's Kappa.
- **`i_CaseStudy/`**: Scripts for case studies on real-world models (e.g., PixelCNN).

---

## **Requirements**

### **Operating System**
Tested on:
- Ubuntu 20.04 LTS or later
- CentOS 7+ (for HPC environments such as Compute Canada)

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
- HPC access (e.g., Compute Canada) for large-scale execution

### **Software Requirements**

- **Python Version:** 3.8 or later  
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

## **Usage: Whole Dataset vs. Sample Data**

**Important:**  
- Running the **whole dataset** requires significant computational resources and time.
- Running the **sample data** is **recommended**, as it provides a quick and effective way to verify the framework's functionality.

---

### **1. Data Collection**

1. **Download Original DNN Programs**:  
   - Download the 60 original DNN programs from StackOverflow:  
     [Download Link](https://bit.ly/3CQPozK).
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

## **Usage: Sample Data**

The `0_Artifact_Testing` directory provides all necessary artifacts to reproduce case study results with minimal computational overhead. It includes:

- **Data**
- **Pre-trained models**
- **Evaluation scripts**
- **Configuration files**
- **Requirements**
- **README instructions**

---

## **Quick Start Guide**

```bash
# Run a sample evaluation
cd 0_Artifact_Testing/scripts
python testForCaseStudy_RCA.py

# Train the fault detection model
cd d_DEFault/A_Detection
python Fault_Detection.py
```

---

## **Licensing Information**

This project is licensed under the **MIT License**, allowing free usage, modification, and distribution. We encourage collaboration and knowledge sharing. See `LICENSE` for more details.
