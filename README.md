# Credit_Risk_Analysis
Evaluate and build supervised machine learning models using scikit-learn and imbalanced-learn to predict credit risk.
## Overview of Analysis

The analysis uses credit card credit dataset from LendingClub.  The dataset is unbalanced as the ratio of good loans far outnumbers the number of risky loans. Therefore, different techniques to train and evaluate models with unbalanced classes are used to predict risk.  The following Python scikit-learn and imbalanced-learn resampling models are compared:

 - Oversampling is first attempted with **RandomOverSampler** and **Smote**.

- **ClusterCentroids** algorithm is used for undersampling.

- An approach that combines undersampling and oversampling:  **SMOTEENN** algorithm. 

- Finally using the imblearn.ensemble library, two ensemble classifiers, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier** are evaluated.

A confusion matrix and imbalanced classification report are generated for all six machine learning models.

### Tools
Jupyter Notebook, Python 3.7.11, Pandas, Numpy, scikit-learn, imbalanced-learn

## Results
After cleaning up the dataset, there were 68,470 loan applicants considered "Low Risk" and 347 considered "High Risk". Less than 1% of loans are classified as "High Risk".

The training and testing datasets were split using train_test_split into 75/25 percent ratio.

And then six resampling machine learning models were trained and used to predict high/low risk.

Balanced Accuracy Scores, Precision and Recall Scores for each model are presented below:

### RandomOverSampler
- Balanced Accuracy Score: **.649**
- Precision Score: 
  - High Risk = 1%
  - Low Risk = ~100%
- Recall Score:
  - High Risk = 63%
  - Low Risk = 67%

![Confusion](/Resources/randomSamplingConfusion.png)
![Classification](/Resources/randomSamplingClassification.png)
---
### SMOTE Oversampling
- Balanced Accuracy Score: **.608**
- Precision Score: 
  - High Risk = 1%
  - Low Risk = ~100%
- Recall Score:
  - High Risk = 55%
  - Low Risk = 66%

![Confusion](/Resources/SmoteConfusion.png)
![Classification](/Resources/SmoteClassification.png)
---
### ClusterCentroids Undersampling
- Balanced Accuracy Score: **.510**
- Precision Score: 
  - High Risk = 1%
  - Low Risk = ~100%
- Recall Score:
  - High Risk = 57%
  - Low Risk = 47%

![Confusion](/Resources/ClusterCentroidsConfusion.png)
![Classification](/Resources/ClusterCentroidsClassification.png)
---
### SMOTEENN Combination (Over and Under) Sampling
- Balanced Accuracy Score: **.640**
- Precision Score: 
  - High Risk = 1%
  - Low Risk = ~100%
- Recall Score:
  - High Risk = 70%
  - Low Risk = 58%

![Confusion](/Resources/SmoteennConfusion.png)
![Classification](/Resources/SmoteennClassification.png)
---

### Balanced Random Forest Ensemble Alogorithm
- Balanced Accuracy Score: **.788**
- Precision Score: 
  - High Risk = 4%
  - Low Risk = ~100%
- Recall Score:
  - High Risk = 67%
  - Low Risk = 91%

![Confusion](/Resources/BalancedForestConfusion.png)
![Classification](/Resources/BalancedForestClassification.png)
---

### Easy Ensemble AdaBoost Alogorithm
- Balanced Accuracy Score: **.925**
- Precision Score: 
  - High Risk = 7%
  - Low Risk = ~100%
- Recall Score:
  - High Risk = 91%
  - Low Risk = 94%

![Confusion](/Resources/EasyEnsembleConfusion.png)
![Classification](/Resources/EasyEnsembleClassification.png)
---
