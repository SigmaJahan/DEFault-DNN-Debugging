# DEFault: A Framework for Fault Detection and Diagnosis in Deep Neural Networks

## Welcome
Welcome to DEFault, a framework designed for detecting and diagnosing faults in deep neural networks. This repository serves as a comprehensive resource for researchers and practitioners looking to leverage the capabilities of DEFault in their own projects.

## Table of Contents
1. [Introduction](#introduction)
2. [How DEFault Works](#how-default-works)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [How to Cite](#how-to-cite)
7. [Authors](#authors)
8. [License](#license)

## Introduction
DEFault consists of three primary stages: 
- **Fault Detection**: Identifying faults within the deep neural network.
- **Fault Categorization**: Classifying the detected faults to understand their nature and impact.
- **Root Cause Analysis**: Investigating to find the underlying causes of the faults detected.

## How DEFault Works
DEFault operates through a structured methodology:
1. **Fault Detection**: Utilizing various algorithms to detect deviations in the model's performance.
2. **Fault Categorization**: Once a fault is detected, the system categorizes it into predefined classes, enabling easier analysis and response.
3. **Root Cause Analysis**: The framework dives deep into the detected and categorized faults to identify the source of the issues, providing insights for remediation.

## Repository Structure
The structure of the repository is designed for ease of access and clarity:
- **0_Artifact_Testing**: Testing artifacts and models developed during the project.
- **1_Dataset**: All datasets used for training and testing.
- **2_Code**: Source code for implementing and running the framework.
- **3_Results**: Results and visualizations from the experiments.
- **4_Experiment_Setup**: Instructions and configuration files for setting up experiments.
- **K_Pre-Print.pdf**: A comprehensive document detailing the research findings.

## Requirements
### Operating System
- Ubuntu
- Windows WSL2
- macOS

### Hardware Requirements
- **Minimum**: 
  - RAM: 8GB
  - Processor: Intel i5 or equivalent

- **Recommended**: 
  - RAM: 16GB
  - Processor: Intel i7 or equivalent

### Software Requirements
- Python 3.10.16
- Virtual environment setup:
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # On macOS/Linux
  venv\Scripts\activate  # On Windows
  ```

## Usage
### Quick Start
To get started quickly with DEFault, check out this Google Colab badge: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) 
Additionally, for an example, see the PixelCNN case study under the `4_Experiment_Setup` directory.

### Running the Complete Experiment
To run the complete experiment, follow these steps:
1. Download the dataset from the following links:
   - [Dataset Link 1](#)
   - [Dataset Link 2](#)
2. Setup the appropriate environment as specified in the Requirements section.
3. Execute the provided scripts found in the `2_Code` directory.

## How to Cite
If you use DEFault in your research, please cite us using the following BibTeX:
```
@inproceedings{YourReferenceHere,
  title={DEFault: A Framework for Fault Detection and Diagnosis in Deep Neural Networks},
  author={First Author and Second Author and Third Author and Fourth Author},
  booktitle={Proceedings of the ICSE 2025},
  year={2025},
}
```

## Authors
- First Author, Dalhousie University, email@example.com
- Second Author, Dalhousie University, email@example.com
- Third Author, Dalhousie University, email@example.com
- Fourth Author, Dalhousie University, email@example.com

## License
This project is licensed under the MIT License. See the LICENSE file for details.