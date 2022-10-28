#!/usr/bin/env python
# coding: utf-8

# # Validation
# 
# Once our model is built and trained, validation is the next phase (we called it testing in the examples).
# 
# 

# ## K-Fold Cross Validation
# 
# >Cross-validation is a statistical method used to estimate the skill of machine learning models. It is commonly used in applied machine learning to compare and select a model for a given predictive modeling problem because it is easy to understand, easy to implement, and results in skill estimates that generally have a lower bias than other methods.
# 
# 
# Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
# 
# The procedure has a single parameter called **k** that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called **k-fold cross-validation**. When a specific value for **k** is chosen, it may be used in place of **k** in the reference to the model, such as **k=10** becoming **10-fold cross-validation**.
# 
# Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data that were not used during the training of the model.
# 
# It is a popular method because it is simple to understand and because it generally results in a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split.
# 
# The general procedure is as follows:
# 
# >- Shuffle the dataset randomly.
# >- Split the dataset into k groups
# >  - For each unique group:
# >        - Take the group as a hold out or test data set
# >        - Take the remaining groups as a training data set
# >        - Fit a model on the training set and evaluate it on the test set
# >        - Retain the evaluation score and discard the model
# >   - Summarize the skill of the model using the sample of model evaluation scores
# 
# Importantly, each observation in the data sample is assigned to an individual group and stays in that group for the duration of the procedure. This means that each sample is given the opportunity to be used in the **hold out** set **1** time and used to train the model **k-1** times.
# 
# Each of these hold outs is called a fold, hence the name of the method
# 
# ```{admonition} Quote
# "This approach involves randomly dividing the set of observations into k groups, or folds, of approximately equal size. The first fold is treated as a validation set, and the method is fit on the remaining k − 1 folds."
# 
# [An Introduction to Statistical Learning: with Applications in R (2021)](http://54.243.252.9/ce-5319-webroot/3-Readings/IntoductiontoStatisticalLearning.pdf)
# 
# ```
# 
# It is also important that any preparation of the data prior to fitting the model occur on the CV-assigned training dataset within the loop rather than on the broader data set. This also applies to any tuning of hyperparameters. A failure to perform these operations within the loop may result in data leakage and an optimistic estimate of the model skill.

# In[ ]:




