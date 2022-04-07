# A-Data-Science-Approach-Towards-Early-Detection-of-Parkinson-s-Disease

This repository explores the problem of early detection of Parkinson's disease using Data Science and Machine Learning ideas, considering numerical speech feature dataset.

Each of he steps involved in a Data Science approach towards solving the problem is explored in this repository, including data preprocessing, exploratory data analysis, feature selection, class imbalance handling, ML based predictive modeling, and hyperparameter tuning.

For the purpose of experimentation, a publicly avalable dataset consisting of Mel Frequency Cepstrum Coefficient(MFCC) based speech features is considered which is cited below-

Sakar, C., Serbes, Gorkem, Gunduz, Aysegul, Nizam, Hatice & Sakar, Betul. (2018). Parkinson's Disease Classification. UCI Machine Learning Repository. (https://archive-beta.ics.uci.edu/ml/datasets/parkinson+s+disease+classification)

The dataset consists of 756 samples with 754 features each. A total of 564 samples are Parkinson's positive and 192 samples are Parkinson's negative, showing clear imbalance in the class distribution.

Firstly, in order to establish a baseline set of results, a simple off the shelf ML classification notebook is presented with filename: Parkinson's_Baseline_ML_Classification.ipynb

From experimental results, we can observe that without any specific data preprocessing or other steps, Gradient Boosting Classifier achieves the best average results on 50 different random train-test splits. The results are-

1. Average Accuracy: 0.8844736842105263
2. Average Sensitivity: 0.8843616793711168
3. Average Specificity: 0.8903755180869922
4. Average AUC: 0.8010664851372816

After this we will include more advanced analysis and techniques, in order to improve the prediction results even further.
