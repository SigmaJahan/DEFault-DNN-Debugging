## Replication Package for DEFault

This repository contains the code and data to replicate the experiments in the paper "Improved Detection and Diagnosis of Faults in  Deep Neural Networks using Hierarchical and Explainable Classification".

### Contents

-   `a_Data_Collection`: Contains the collected and filtered StackOverflow posts used to create the dataset.
    
-   `b_Fault_Seeding`: Contains the code to inject faults into DNN models using an extended version of DeepCrime.
    
    -   `Part 1-DC`: Original DeepCrime code.
        
    -   `Part 2-EFI`: Our extensions to support more fault types and RNN models.
        
-   `c_Feature_Extraction`: Code to extract static and dynamic features from DNN models.
    
-   `d_DEFault`: Implementation of the DEFault approach, including fault detection, diagnosis, and explanation models.
    
-   `e_Evaluation`: Code to evaluate DEFault on real-world and seeded faults. Sample test data included.
    
-   `f_Figures`: Generated figures from the paper.
    
-   `g_Dataset`: Labelled dataset used to train the DEFault models. Includes dynamic features and fault labels.
    
-   `h_CohenKappaAnalysis`: Cohen's kappa analysis to validate dataset creation.

### Usage
To replicate the experiments:
1.  Data Collection and Preprocessing
    
    -   Run the Jupyter notebook in  `c_Feature_Extraction/Dynamic`  to extract dynamic features and generate the dataset in  `g_Dataset/labeled_dynamic_feature_first_level`.
    -   Run  `c_Feature_Extraction/Static/Static_Feature_Extraction.py`  to extract static features.
    -   Concatenate static features to dataset in  `g_Dataset/labeled_dynamic_feature_second_level`.
2.  Model Training
    -   Train the Level 1 fault detection model using  `d_DEFault/Fault_Detection.py`.
    -   Train the Level 2 fault categorization models for each category (activation, layer, loss etc.) using scripts in  `d_DEFault/`.
    -   Train the explanation model for layer faults using  `d_DEFault/Fault_Explainer_Framework.py`.
3.  Model Evaluation
    -   Run  `e_Evaluation/Fault_Evaluation_Detection_Diagnosis.py`  to evaluate the trained models on real-world and seeded faults.
    -   Use the test data in  `e_Evaluation/testData/`  for evaluation.
    -   Generate classification reports, confusion matrices etc. to summarize results.

### Requirements
See  `requirements.txt`  for Python library requirements.

### Licensing Information
This project is licensed under the MIT License, a permissive open-source license that allows others to use, modify, and distribute the project's code with very few restrictions. This license can benefit research by promoting collaboration and encouraging the sharing of ideas and knowledge. With this license, researchers can build on existing code to create new tools, experiments, or projects, and easily adapt and customize the code to suit their specific research needs without worrying about legal implications. The open-source nature of the MIT License can help foster a collaborative research community, leading to faster innovation and progress in their respective fields. Additionally, the license can help increase the visibility and adoption of the project, attracting more researchers to use and contribute to it.