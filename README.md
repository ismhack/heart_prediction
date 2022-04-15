Heart Disease Prediction using Machine Learning algorithms


Contents

1 Introduction 3 2 Related Work 3 3 Development Plan 3 4 Problem Description 3

1. Data Set . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1. Prediction Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

5 Results 5

1. Logistic Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
1. Decision Tree . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
1. Random Forest . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
1. Artificial Neural Network (ANN) . . . . . . . . . . . . . . . . . . . . . . . . . 10

6 Conclusions 11

1. Outcome comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
1. Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
1  Introduction

Based on the Center for Disease Control and Prevention, heart disease is the leading cause of death for men, women, and people of most racial and ethnic groups in the United States. One person dies every 36 seconds in the United States from cardiovascular disease. About 655,000 Americans die from heart disease each year, that’s 1 in every 4 deaths. Heart disease costs the United States about $219 billion each year from 2014 to 2015. This includes the cost of health care services, medicines, and lost productivity due to death [4].

This makes heart disease a major concern to be dealt with. But it is difficult to identify heart disease because of several risks factors such as age, high blood pressure, high cholesterol and many other. This is the reason researchers have turn their attention to new methods to model these kinds of problems like Machine Learning algorithms.

Machine learning(ML) model predictions allow businesses to make highly accurate guesses as to the likely outcomes of a question based on historical data. There are multiple heart datasets available on Internet. For this project I have selected the Clevelant Heart Disease dataset[6] since it is the most refined and there is a wide range of results posted online which can be used to compare against the output of this project.

The purpose of this project is to explore existing papers related to solve heart disease prediction problem and to experiment with the top most efficient ML algorithms in search of improvement opportunities. Compare those algorithms and provide an insight of the accuracy, sensitivity and efficiency towards this problem.

2  Related Work

There are articles that study the prediction of heart diseases using Artificial Neuronal Net- works (ANN), logistic regression, decision tree, ensemble model and others. In [7] the author proposes a prediction model based on ensemble, which combines three independent models Support Vector Machine (SVM), decision tree and ANN to improve the accuracy of each independent model to 87%. The authors in [2] discuss an approach to select the best features from the dataset using Principal Component Analysis (PCA), and apply them to a logistic regression algorithm on which tuning techniques have been implemented and resulted in 100% accuracy.

3  Development Plan

The development plan of this project will consist of 3 phases.

Research

This phase is to acquire current state of the problem, learn about the progress made in this field and the methodologies utilized to solve this or similar tasks. As well as to review datasets and practises to gain knowledge, and use it towards this project.

Estimated completion date is Oct 25, 2020.

Implementation

After the research is done, the information collected would serve as a guide of what ML al- gorithms and techniques to implement and experiment with. In this phase, the motivation is to try a variation of techniques, parameters or combinations to pursuit improvement in the existing algorithms accuracy and sensitivity.

Estimated completion date is Nov, 10 2020.

Analyze and Report

In the last phase, the results of the experiment will be analyzed: a detailed comparison of the algorithms implemented to show their effectiveness and derive their properties. Estimated completion date is Nov 18, 2020.

4  Problem Description

ML algorithms can assist us in making decisions and predictions based on mathematical models. ML offers a variety of techniques and algorithms that have different properties and can work more adequately on one type of problem, but the same technique might not work on a different problem. In this project, I will be applying different ML approaches to the heart disease dataset, and compare them using evaluation metrics and classification accuracy.

1. Data Set

The dataset used in this article is the Cleveland Heart Disease dataset taken from the UCI repository. This database contains 76 attributes, but all published experiments refer to us- ing a subset of 14 of them. This subset contains the most important predictors and factors to determine the disease. For example, and to mention few of these features. Age: Studies indicate that the risk of stroke doubles every decade after age 55. Other studies show that men have higher risk of coronary disease, however females with diabetes is more likely to develop a heart disease than men. High blood pressure overtime damages the arteries in

the human body, in combine with diabetes or high cholesterol the chances of developing a heat disease are even higher. The objective of the project is not to refine the features on the dataset, but to focus on the problem using work from previous papers. Then analyze the result given by different ML algorithms and to strive for areas of improvement and result which can then be leveraged in future works.

Below is detailed description of each feature:

1. Age of the patient in years (AGE).
1. Sex of the patient (SEX). 1 = male; 0 = female
1. Chest Pain Type (CP). 1 = typical angina; 2= atypical angina; 3= non-anginal pain; 4= asymptomatic
1. Resting Blood Pressure (TRESTBPS) in mmHg.
1. Serum Cholesterol in mg/dl (CHOL).
1. Fasting Blood Sugar greater than 120 mg/dl (FBS). 1 = true; 0 = false
1. Resting Electrocardiograph results(REST\_ECG). 0 = normal; 1 = having STT wave abnormality; 2 = showing probable or definite left ventricular hypertrophy by Estes criteria
1. Maximum Heart Rate Achieved in bps(THAL\_ACH).
1. Exercise Induced Angina (EXANG). 1 = yes; 0 = no
1. ST depression induced by exercise relative to rest (OLD\_PEAK).
1. The slope of the peak exercise ST segment (SLOPE). 1 = upsloping; 2 = flat; 3 = downsloping
1. Number of major vessels (0-3) colored by flourosopy (CA).
1. Thalassemia (THAL). 3 = normal; 6 = fixed defect; 7 = reversible defect
1. The predicted attribute (PRED). 1 = true; 0 = false
2. Prediction Models

As stated in previous sections, this project will consist in applying supervised ML models, evaluate them and intend to derive their properties in search of improving their accuracy. I will be implementing these ML models in python using the sklearn[3] and keras[5] libraries. The following classification models will be implemented:

- Logistic Regression is a type of regression analysis in statistics used for prediction which uses a logistic function to produce an output between 0 and 1.

1 logistic(z) = ![](REPORT/Aspose.Words.b5baeff1-1b8f-40f4-b98d-0d0119af3f93.001.png)

1 + exp(-z)
