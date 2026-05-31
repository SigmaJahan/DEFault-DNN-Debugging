# Evaluation of DEFault on a sample data

**Note:** This package focuses on reproducing the case study results (included in the paper) due to the computational and time-intensive nature of processing the entire dataset. To reduce computational overhead, we provided extracted dynamic and static features from the sample PixelCNN model. These pre-processed features allow the scripts to run efficiently while producing the same results as those discussed in the paper.

---
## Directory Explanation
- **Evaluation Scripts**: Python scripts for Fault Detection, Fault Categorization, and Root Cause Analysis.
- **Models**: Pre-trained models saved using `joblib`. These include classifiers for specific fault types (e.g., loss function faults, layer faults).
- **Data**: Sample datasets required to reproduce the case study results, including dynamic and static features extracted from the PixelCNN model. [PixelCNN Model](https://github.com/sarus-tech/tf2-published-models)
- **Config**: Configuration files to customize parameters for execution.
---


