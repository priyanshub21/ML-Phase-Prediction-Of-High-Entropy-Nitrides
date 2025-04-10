# README for High-Entropy Nitrides (HENs) Machine Learning Project

## Project Overview
This project focuses on the development of machine learning models to predict the structural stability and classify the phases of High-Entropy Nitrides (HENs). HENs are materials with exceptional properties such as high hardness, thermal stability, and corrosion resistance, making them suitable for various applications. The complexity of their compositional space presents challenges in phase classification and stability prediction, which this project aims to address using advanced machine learning techniques.

## Objectives
1. Develop machine learning models for phase classification of high-entropy nitrides.
2. Create predictors for the structural stability of HENs to facilitate the discovery of new high-entropy materials.

## Properties of High-Entropy Nitrides
- High hardness
- Thermal stability
- Corrosion resistance
- Dependence on thermodynamic and physical factors such as configurational entropy, enthalpy of formation, atomic size mismatch, and electronic structure.

## Sorting Criteria for Compositions
- For quaternary and quinary compositions:
  - **1.34 ≤ Xp ≤ 1.94**
  - **1.89 ≤ Prad ≤ 2.54**
  - **-2.72 ≤ ΔHmix ≤ 0.76**

## Approach
1. **Dataset Generation**: Created a semi-synthetic dataset using atomic environment mapping based on existing datasets.
2. **Sorting Criteria**: Developed distinct sorting criteria for quinary and quaternary compositions to identify candidates for single-phase classification.
3. **ADASYN Implementation**: Utilized the Adaptive Synthetic Sampling Approach for Imbalanced Learning (ADASYN) to generate a synthetic dataset from structural modeling and literature to oversample the minority class.
4. **Machine Learning Algorithms**: Implemented and trained four machine learning algorithms:
   - K-Nearest Neighbors (KNN)
   - Random Forest (RF)
   - Support Vector Machine (SVM)
   - Gaussian Naive Bayes (GNB)
5. **Feature Pool Design**: Designed a feature pool based on structural and thermodynamic parameters.
6. **Evaluation Metrics**: Used correlation matrix, learning curves, accuracy metrics, confusion matrix, and ROC-AUC curves for model evaluation.

## Results
- The KNN model achieved an accuracy of **99.63%** on training data and **93.4%** on test data after cross-validation.
- After implementing ADASYN, the KNN model achieved an accuracy of **96.7%** on training data and **94.2%** on test data.

## Dataset
The dataset used for this project is available in the `dataset.csv` file. This file contains the features and labels necessary for training and testing the machine learning models.

### Dataset Structure
- **Columns**: The dataset includes various features related to the structural and thermodynamic properties of HENs, as well as the target variable indicating the phase classification.
- **Format**: CSV (Comma-Separated Values)

## Instructions for Use
1. **Prerequisites**: Ensure you have Python installed along with the necessary libraries:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn

2. **Clone the Repository**: Clone this repository to your local machine.
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. **Load the Dataset**: Use the following code snippet to load the dataset:
   ```python
   import pandas as pd

   # Load the dataset
   data = pd.read_csv('dataset.csv')
   ```

4. **Train the Models**: Follow the provided scripts to train the machine learning models. Ensure to preprocess the data as necessary, including handling imbalanced classes using ADASYN.

5. **Evaluate the Models**: Use the evaluation metrics provided in the project to assess the performance of the models.

6. **Visualize Results**: Utilize the provided visualization scripts to generate plots such as confusion matrices, ROC-AUC curves, and learning curves.

## Conclusion
This project demonstrates the potential of machine learning in predicting the structural stability and phase classification of High-Entropy Nitrides. The developed models can significantly accelerate material discovery and guide experimental research in this field.

## Acknowledgments
Thank you for your interest in this project! For any questions or contributions, please feel free to reach out.

---

