## Replication Package for DEFault

This repository contains the code and data to replicate the experiments in the paper "Improved Detection and Diagnosis of Faults in Deep Neural Networks using Hierarchical and Explainable Classification".

### Contents

- `a_Data_Collection`: Contains the collected and filtered StackOverflow posts used to create the dataset.

- `b_Fault_Seeding`: Contains the code to inject faults into DNN models using an extended version of DeepCrime.

  - `Part 1-DC`: Original DeepCrime code.
  - `Part 2-EFI`: Our extensions to support more fault types, including convolutional and recurrent models.

- `c_Feature_Extraction`: Code to extract static and dynamic features from DNN models.

  - `Dynamic`: Scripts for extracting runtime (dynamic) features.
  - `Static`: Scripts for extracting model structure (static) features.

- `d_DEFault`: Implementation of the DEFault approach, including hierarchical fault detection, diagnosis, and explainability models.

  - `A_Detection`: Scripts for fault detection.
  - `B_Categorization`: Scripts for fault categorization into subtypes (e.g., activation, layer, loss).
  - `C_RootCauseAnalysis`: Explainability scripts for root cause analysis.

- `e_Evaluation`: Code to evaluate DEFault on real-world and seeded faults. Includes benchmark data and analysis.

- `f_Figures`: Figures generated for the paper (e.g., fault hierarchy, dataset distributions).

- `g_Dataset`: Labeled datasets for training and evaluation of DEFault models.

  - `labeled_dynamic_feature_first_level`: Dynamic features for initial fault detection.
  - `labeled_dynamic_feature_second_level`: Features categorized by fault types (e.g., layer, activation).
  - `labeled_dynamic_feature_third_level`: Additional dataset for detailed root cause analysis.
  - `labeled_static_feature_RCA`: Static features used for root cause analysis.

- `h_CohenKappaAnalysis`: Scripts and results for validating dataset consistency using Cohen's kappa.

- `i_CaseStudy`: Additional case studies and extracted features for testing DEFault on specific DNN programs (e.g., PixelCNN).

### Usage

To replicate the experiments:

1. **Data Collection and Preprocessing**
   - Run the Jupyter notebook in `c_Feature_Extraction/Dynamic` to extract dynamic features and generate the dataset in `g_Dataset/labeled_dynamic_feature_first_level`.
   - Run `c_Feature_Extraction/Static/Static_Feature_Extraction.py` to extract static features.
   - Concatenate static features to the dataset in `g_Dataset/labeled_dynamic_feature_second_level`.

2. **Model Training**
   - Train the Level 1 fault detection model using `d_DEFault/A_Detection/Fault_Detection.py`.
   - Train the Level 2 fault categorization models using scripts in `d_DEFault/B_Categorization`.
   - Train root cause analysis models using `d_DEFault/C_RootCauseAnalysis/RCA-*.py`.

3. **Model Evaluation**
   - Run `e_Evaluation/Fault_Evaluation_Detection_Diagnosis.py` to evaluate the trained models on real-world and seeded faults.
   - Use the test data in `e_Evaluation` for evaluation.
   - Generate classification reports, confusion matrices, and other evaluation metrics.

4. **Case Studies**
   - Test DEFault on case studies (e.g., PixelCNN) using scripts in `i_CaseStudy`.
   - Analyze dynamic and static features extracted for case studies.

### Requirements

See `requirements.txt` for Python library requirements.

### Licensing Information

This project is licensed under the MIT License, a permissive open-source license that allows others to use, modify, and distribute the project's code with very few restrictions. This fosters collaboration, encourages knowledge sharing, and supports the adaptation of the code for new research.
