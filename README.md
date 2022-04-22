# Credit_Risk_Analysis
Evaluate and build machine learning models using scikit-learn to predict credit risk.
## Overview of Analysis
Using python and scikit-learn, this project evaluates machine learning models by using resampling to determine which is better at predicting credit risk.

Oversampling is first attempted with RandomOverSampler and Smote.

ClusterCentroids algorithm is used for undersampling.

An approach that combines undersampling and oversampling is then run against the SMOTEENN algorithm. 

Finally using the imblearn.ensemble library, two ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier are used to predict credit risk.

