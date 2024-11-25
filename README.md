# GasTurbine-FDD
This repository contains the implementation of a machine learning model for Fault Detection and Diagnosis (FDD), specifically developed for gas turbines as described in the paper:

"Fault detection and diagnosis for industrial processes based on clustering and autoencoders: a case of gas turbines"

Authors: Jose M. Barrera, Alejandro Reina, Alejandro Mate, Juan C. Trujillo
Published in: International Journal of Machine Learning and Cybernetics, 13(10), 3113-3129 (2022)
DOI: ([10.1007/s13042-022-01583-x](https://link.springer.com/article/10.1007/s13042-022-01583-x))

## Overview
This project presents a novel machine learning architecture for Fault Detection and Diagnosis (FDD) in complex industrial processes. It combines clustering techniques to identify operational stages and autoencoders trained on normal operational data to detect anomalies. The model includes:
  
1. Detect different stages in an industrial process using clustering techniques.
2. Identify faults in real time using autoencoders trained on normal operational data.
3. Dampen false positives with a sliding window mechanism

## Repository Structure
- `AE/`: Contains the weights of the autoencoders trained for each operational stage of the gas turbine.
  - `cluster0/`
  - `cluster1/`
  - `cluster2/`
  - `cluster3/`
  - `cluster4/`
  - `cluster5/`

- `Clasificador/`: Includes the implementation of the classifier used to assign incoming data tuples to the appropriate operational stage.

- `Scalers/`: Stores the scalers (normalizers) required for the project:
  - Normalizers for the classifier.
  - Normalizers for each autoencoder corresponding to the operational stages.

- `predictor.py`: A script that combines all components (clustering, autoencoders, and sliding window) to predict anomalies in real-time.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fault-detection-diagnosis.git
   

## Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow`/`keras`
  - `matplotlib`

## Citation

If you use this repository in your research or work, please cite the original paper:

**ISO 690 Format:**
BARRERA, Jose M., et al. Fault detection and diagnosis for industrial processes based on clustering and autoencoders: a case of gas turbines. International Journal of Machine Learning and Cybernetics, 2022, vol. 13, no 10, p. 3113-3129.

**BibTeX:**
```bibtex
@article{barrera2022fault,
  title={Fault detection and diagnosis for industrial processes based on clustering and autoencoders: a case of gas turbines},
  author={Barrera, Jose M and Reina, Alejandro and Mate, Alejandro and Trujillo, Juan C},
  journal={International Journal of Machine Learning and Cybernetics},
  volume={13},
  number={10},
  pages={3113--3129},
  year={2022},
  publisher={Springer}
}

   
