# A-Data-Science-Approach-Towards-Early-Detection-of-Parkinson-s-Disease

This repository explores the problem of early detection of Parkinson's disease using Data Science and Machine Learning ideas, considering numerical speech feature dataset.

Each of he steps involved in a Data Science approach towards solving the problem is explored in this repository, including data preprocessing, exploratory data analysis, feature selection, class imbalance handling, ML based predictive modeling, and hyperparameter tuning.

For the purpose of experimentation, a publicly avalable dataset consisting of Mel Frequency Cepstrum Coefficient(MFCC) based speech features is considered which is cited below-

Sakar, C., Serbes, Gorkem, Gunduz, Aysegul, Nizam, Hatice & Sakar, Betul. (2018). Parkinson's Disease Classification. UCI Machine Learning Repository. (https://archive-beta.ics.uci.edu/ml/datasets/parkinson+s+disease+classification)

The dataset consists of 756 samples with 754 features each. A total of 564 samples are Parkinson's positive and 192 samples are Parkinson's negative, showing clear imbalance in the class distribution.

### Baseline ML Classification Results

Firstly, in order to establish a baseline set of results, a simple off the shelf ML classification notebook is presented with filename: Parkinson's_Baseline_ML_Classification.ipynb

From experimental results, we can observe that without any specific data preprocessing or other steps, Gradient Boosting Classifier achieves the best average results on 50 different random train-test splits. The results are-

1. Average Accuracy: 0.8844736842105263
2. Average Sensitivity: 0.8843616793711168
3. Average Specificity: 0.8903755180869922
4. Average AUC: 0.8010664851372816

After this we will include more advanced analysis and techniques, in order to improve the prediction results even further.

### Studying the impact of Feature Scaling and Selection

We next study the impact of Feature Scaling and Selection on the overall classification performance. For this, firstly a baseline is established using an off the shelf GBC classifier. Next MinMaxScaler and StandardScaler approaches are studied. Finally Recursive Feature Elimination with Cross-Validation (RFECV) is studied. The average test-set results for each of these are given below.

GBC Baseline Results

1. Average Accuracy:  0.8855263157894736
2. Average Sensitivity:  0.8858855026650322
3. Average Specificity:  0.8874589753822438
4. Average AUC:  0.8036215112321307

GBC-MinMaxScaler Results

1. Average Accuracy:  0.8817105263157895
2. Average Sensitivity:  0.8833005396240781
3. Average Specificity:  0.8788876466923539
4. Average AUC:  0.7983685046516906

GBC-StandardScaler Results

1. Average Accuracy:  0.8810526315789474
2. Average Sensitivity:  0.8825574769857225
3. Average Specificity:  0.8798134931004691
4. Average AUC:  0.7970864533696391

GBC-RFECV Results

1. Average Accuracy:  0.8828947368421054
2. Average Sensitivity:  0.8881704989373824
3. Average Specificity:  0.8650476031007328
4. Average AUC:  0.8053778080326753

From these experimental results it is clear that feature scaling is definitely not helping.

Feature selection does improve the Avg. AUC and Sensitivity very slightly, however it decreases the Avg. Accuracy and Specificity in a similar way.

So far none of these approaches seems to be effective at improving the overall classification performances.
