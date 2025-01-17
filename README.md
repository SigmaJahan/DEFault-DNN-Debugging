# **Replication Package for DEFault**

This repository contains the code and data to replicate the experiments from the paper:  
**"Improved Detection and Diagnosis of Faults in Deep Neural Networks using Hierarchical and Explainable Classification."**

DEFault combines static and dynamic analyses to detect, categorize, and diagnose faults in Deep Neural Network (DNN) programs, addressing limitations in existing techniques.

---

## **Table of Contents**
1. [Contents](#contents)
2. [Requirements](#requirements)
3. [Usage](#usage)
   - [1. Data Collection](#1-data-collection)
   - [2. Feature Extraction](#2-feature-extraction)
   - [3. Model Training](#3-model-training)
   - [4. Model Evaluation](#4-model-evaluation)
   - [5. Case Studies](#5-case-studies)
4. [Licensing Information](#licensing-information)

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
  - **A_Detection**: Fault detection scripts.
  - **B_Categorization**: Fault categorization scripts.
  - **C_RootCauseAnalysis**: Root cause analysis scripts.
- **`e_Evaluation/`**: Scripts to evaluate DEFault on real-world and seeded faults.
- **`f_Figures/`**: Figures used in the paper.
- **`g_Dataset/`**: Labeled datasets for training and evaluation:
  - **Dynamic Features**: For fault detection and categorization.
  - **Static Features**: For root cause analysis.
- **`h_CohenKappaAnalysis/`**: Scripts for dataset consistency validation using Cohen's Kappa.
- **`i_CaseStudy/`**: Scripts for case studies on real-world models (e.g., PixelCNN).

---

## **Requirements**
### **Hardware**
- **Minimum Requirements**:
  - CPU: 4 cores
  - RAM: 8 GB
  - Disk: 10 GB
- **Recommended**:
  - GPU: NVIDIA with CUDA support for faster execution.

### **Software**
- **Python Version**: 3.8 or later  
- **Dependencies**: Listed in `requirements.txt`. Install using:  
  ```bash
  pip install -r requirements.txt
  ```

---

## **Usage**

### **1. Data Collection**
1. **Download Original DNN Programs**:
   - Download the 60 original DNN programs from StackOverflow:
     [Download Link](https://bit.ly/3CQPozK).
2. **Inject Faults**:
   - Use the extended DeepCrime framework to generate a dataset of ~14.5K DNN programs:
     ```bash
     cd b_Fault_Seeding
     python Fault_Seeding_Script.py
     ```

---

### **2. Feature Extraction**
1. **Extract Static Features**:
   ```bash
   cd c_Feature_Extraction/Static
   python Static_Feature_Extraction.py
   ```
2. **Extract Dynamic Features**:
   ```bash
   cd c_Feature_Extraction/Dynamic
   python Dynamic_Feature_Extraction.py
   ```

---

### **3. Model Training**
1. Train the Level 1 fault detection model:
   ```bash
   cd d_DEFault/A_Detection
   python Fault_Detection.py
   ```
2. Train the Level 2 fault categorization models:
   ```bash
   cd d_DEFault/B_Categorization
   python Fault_Categorization.py
   ```
3. Train the Level 3 root cause analysis models:
   ```bash
   cd d_DEFault/C_RootCauseAnalysis
   python RCA_Analysis.py
   ```

---

### **4. Model Evaluation**
1. Evaluate DEFault on the dataset and real-world DNN programs:
   ```bash
   cd e_Evaluation
   python Fault_Evaluation_Detection_Diagnosis.py
   ```
2. **Outputs**:
   - Metrics (accuracy, precision, recall, F1-scores).
   - Confusion matrices and classification reports.

---

### **5. Case Studies**
Analyze faults in real-world models like PixelCNN:
1. Extract features:
   ```bash
   cd i_CaseStudy
   python Feature_Extraction_CaseStudy.py
   ```
2. Run analysis:
   ```bash
   python PixelCNN_Analysis.py
   ```

---

## **Licensing Information**
This project is licensed under the **MIT License**. You are free to use, modify, and distribute the code with minimal restrictions, fostering collaboration and knowledge sharing.
