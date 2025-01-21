# Case Study Artifacts for ICSE 2025 Artifact Evaluation

## Overview
This repository contains the necessary artifacts to reproduce the case study results presented in the associated research paper. The artifacts provided include data files, pre-trained models, and scripts used for the evaluation.

**Note:** This package focuses on reproducing the case study results (included in the paper) due to the computational and time-intensive nature of processing the entire dataset.

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

## Installation
### Prerequisites
Ensure you have Python 3.10+ installed. You can install the required dependencies using:

```
pip install -r requirements.txt
```

## Configuration
Configuration parameters are stored in `config/config.ini`. Ensure the paths and parameters are correctly set before running the scripts.

## Running the Case Study
To execute the scripts and reproduce the results, follow these steps:

1. Run the FD_FC case study script (Fault Detection and Fault Categorization):
   ```
   python scripts/testForCaseStudy_FD_FC.py 

2. Run the RCA case study script (Root Cause Analysis):
   ```
   python scripts/testForCaseStudy_RCA.py

### Expected Output
The results will be visible on the console for respective scripts and should be comparable to the results presented in the paper.

## Artifacts Explanation
- **Evaluation_Scripts:** Python scripts to execute the case studies.
- **Models:** Pre-trained models saved using `joblib`.
- **Data:** Sample datasets required to reproduce the case study results.
- **Config:** Configuration files to customize parameters for execution.

## Known Limitations
- Full experiment replication requires access to high-performance computing resources (e.g., Compute Canada). The dynamic features processing requires significant computation time and resources. Hence for the sample DNN model, the dynamic feature and static feature (extracted) are provided to run the scripts. 


By following the provided steps and documentation, users should be able to reproduce the case study results effectively (please check the paper's case study section) and gain insights into the methodologies used in the paper.