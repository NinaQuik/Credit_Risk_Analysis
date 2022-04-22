# Credit_Risk_Analysis
Evaluate and build machine learning models using scikit-learn to predict credit risk.
## Overview of Analysis
Using python and scikit-learn, this project evaluates machine learning models by using resampling to determine which is better at predicting credit risk.

Oversampling is first attempted with **RandomOverSampler** and **Smote**.

**ClusterCentroids** algorithm is used for undersampling.

An approach that combines undersampling and oversampling is then run against the **SMOTEENN** algorithm. 

Finally using the imblearn.ensemble library, two ensemble classifiers, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier** are used to predict credit risk.

A confusion matrix and imbalanced classification report is generated for all six machine learning models.

### Tools
Jupyter Notebook, Python 3.7.11, Pandas, Numpy, scikit-learn, imbalanced-learn

## Results
Balanced Accuracy Scores, Precision and Recall Scores for each model.

### RandomOverSampler
- Balanced Accuracy Score: **.649**
- Precision Score: 1.0 percent, high risk, 
- Recall Score:

![Confusion](/Resources/randomSamplingConfusion.png)
![Classification](/Resources/randomSamplingClassification.png)
---

### SMOTE Oversampling
- Balanced Accuracy Score: **.608**
- Precision Score:
- Recall Score: 

![Confusion](/Resources/SmoteConfusion.png)
![Classification](/Resources/SmoteClassification.png)
---
