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

## Summary

All the models were good at low risk precision, however there is a 99.5% chance that a credit is low risk, machine learning doesn't offer much advantage there. Precision of high risk (out of all the examples that were predicted high risk, how many were really high risk) ranged from 1% to 7%. False positives make of over 90% of high risk predictions.

Sensitivity or Recall of high risk predictions (of all the high risk credits, how many were predicted as high risk) varied from from 55% to 91%.  Low risk recall followed similar patterns (47% to 94%).

Of the six different models evaluated, Easy Ensemble AdaBoost had the highest balanced accuracy socre, .925, the highest recall scores (91% high risk, 94% low risk) and the least bad precision score (7% high risk).

CluserCentroids undersampling performed the worst with a balanced accuracy score of .510, and recall scores (57% and 47%). More than half the low risk credit cards were incorrectly labeled as high risk!

If one of the above six machine learning models had to be used for credit risk predictions, Easy Ensemble AdaBoost is the easy winner. However, a precision rate of 7% implies that there will be many low risk good customers incorrectly flagged as high risk.  A recommendation would be to look at more advanced machine learning techniques, ex, neural networks, to see if improvements can be found with other models.
