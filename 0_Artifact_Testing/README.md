# Case Study Artifacts for ICSE 2025 Artifact Evaluation

## Overview
This repository contains the necessary artifacts to reproduce the case study results presented in the associated research paper (Pre-print: https://arxiv.org/abs/2501.12560). The artifacts provided include data files, pre-trained models, and scripts used for the evaluation.

**Note:** This package focuses on reproducing the case study results (included in the paper) due to the computational and time-intensive nature of processing the entire dataset.

---

## Repository Structure
```
0_Artifact_Testing/
├── README.md                 # Documentation (this file)
├── requirements.txt          # Dependencies list
├── config/
│   ├── config.ini            # Configuration settings
├── models/
│   ├── best_clf_weights.joblib
│   ├── best_clf_layer.joblib
│   ├── best_clf_loss.joblib
│   ├── best_clf_regularizer.joblib
│   ├── best_clf_optimization.joblib
│   ├── best_clf_detection.joblib
│   ├── best_clf_activation.joblib
│   ├── best_clf_hyperparameter.joblib
├── evaluation_scripts/
│   ├── testForCaseStudy_RCA.py  # Script for RCA case study
│   ├── testForCaseStudy_FD_FC.py # Script for FD_FC case study
├── data/
│   ├── pixelcnn_buggy.csv  
│   ├── static_features_df_test_file.csv 
└── LICENSE.txt        
```

---

## Installation
### Prerequisites
Ensure you have Python 3.10+ installed. You can install the required dependencies using:

```
pip install -r requirements.txt
```

---

## Configuration
Configuration parameters are stored in `config/config.ini`. Ensure the paths and parameters are correctly set before running the scripts.

---

## Running the Case Study
To execute the scripts and reproduce the results, follow these steps:

1. **Run the FD_FC Case Study Script (Fault Detection and Fault Categorization):**
   Execute the script to analyze the presence and types of faults in the PixelCNN model:
   ```
   python evaluation_scripts/testForCaseStudy_FD_FC.py
   ```
   - This script performs fault detection and categorization.  
   - Expected Results:
     - **Fault Detection**: Confirms the presence of faults in the model.  
     - **Fault Categorization**: Identifies three types of faults:
       - Loss Function Fault
       - Hyperparameter Fault
       - Layer Fault  
     - Note: DEFault may incorrectly identify an additional Optimization Fault due to feature overlap, as discussed in the paper.

2. **Run the RCA Case Study Script (Root Cause Analysis):**
   Execute the script to diagnose the root causes of the detected faults:
   ```
   python evaluation_scripts/testForCaseStudy_RCA.py
   ```
   - This script identifies specific root causes of the Layer Fault using static features.  
   - Expected Results:
     - **Root Cause Analysis (RCA)**: The following root causes will be identified:
       - **Top@1: CountDense**: Check the configuration and number of Dense layers.
       - **Top@2: Max_Neurons**: Verify the maximum number of neurons in any single layer.
       - **Top@3: CountConv2D**: Inspect the configuration of 2D convolutional layers.

---

## Expected Output
Upon running the provided scripts, the following outputs will be generated:
1. **FD_FC Script Outputs**:
   - Fault detection result indicating that faults are present in the model.
   - Fault categorization identifying three faults:
     - Loss Function Fault
     - Hyperparameter Fault
     - Layer Fault  
   - Note: The script may also incorrectly identify Optimization Fault due to overlapping features, as discussed in the paper.

2. **RCA Script Outputs**:
   - Root cause analysis identifying specific root causes:
     - Misconfigured number of Dense layers (`CountDense`).
     - Incorrect neuron count in layers (`Max_Neurons`).
     - Issues with 2D convolutional layers (`CountConv2D`).

---

## Artifacts Explanation
- **Evaluation Scripts**: Python scripts for Fault Detection, Fault Categorization, and Root Cause Analysis.
- **Models**: Pre-trained models saved using `joblib`. These include classifiers for specific fault types (e.g., loss function faults, layer faults).
- **Data**: Sample datasets required to reproduce the case study results, including dynamic and static features extracted from the PixelCNN model.
- **Config**: Configuration files to customize parameters for execution.

---

## Known Limitations
- Full experiment replication requires access to high-performance computing resources (e.g., Compute Canada).  
- To reduce computational overhead, we provide extracted dynamic and static features from the sample PixelCNN model. These pre-processed features allow the scripts to run efficiently while producing the same results as those discussed in the paper.

---

## Verifying the Results
The provided artifacts enable reviewers to reproduce and verify the results of the case study without referring to the paper. Simply run the scripts as instructed above, and compare the generated outputs (fault detection, fault categorization, and root cause analysis) with the expected results outlined in this document.

By following the steps and using the provided artifacts, reviewers can confirm the validity of the methodologies and results presented in the research paper.
