# ImagoAI-ML-Intern

# Mycotoxin Prediction from Hyperspectral Corn Data

## Overview

This repository contains a complete solution for predicting vomitoxin_ppb levels in corn samples using hyperspectral imaging data. The solution encompasses data preprocessing, dimensionality reduction via PCA, and development of a Convolutional Neural Network (CNN) with hyperparameter tuning.

## Repository Structure

- **ml-intern-imagoai.ipynb**  
  A Jupyter Notebook that contains clean, modular, and well-commented code for:
  - Data exploration and preprocessing (including handling missing values and standardization).
  - Dimensionality reduction using PCA.
  - Building and training a CNN with hyperparameter tuning using scikeras and GridSearchCV.
  - Evaluation of the model with performance metrics and visualizations (scatter plot of actual vs. predicted values).

- **report.md** (or **report.pdf**)  
  A short report summarizing:
  - Preprocessing steps and rationale.
  - Insights from dimensionality reduction.
  - Model selection, training, and evaluation details.
  - Key findings and suggestions for future improvements.

- **requirements.txt**  
  A list of all dependencies required to run the code:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - scikeras

- **README.md**  
  This file, containing instructions for installing dependencies and an overview of the repository structure.

## Installation and Running the Code

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   
2. **Create a Virtual Environment:** 
   ```python
   -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install Dependencies** 
   ```
   pip install -r requirements.txt

4. **Run the Notebook** 
   ```
   Launch Jupyter Notebook:
   jupyter notebook ml-intern-imagoai.ipynb
   Alternatively, open the notebook in a Kaggle Notebook environment if preferred.

## Dependencies
  - Python 3.8+
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - scikeras

## Notes
  - Hyperparameter tuning with GridSearchCV for deep learning models is computationally intensive. Consider using a reduced grid or RandomizedSearchCV for faster tuning if necessary.
  - The repository includes a baseline CNN model. Future improvements could include exploring advanced architectures like attention mechanisms or transformer models.
