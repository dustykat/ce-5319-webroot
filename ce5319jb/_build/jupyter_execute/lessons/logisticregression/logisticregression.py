#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# ![](https://d2slcw3kip6qmk.cloudfront.net/marketing/blog/2018Q2/critical-elements-for-decision-making/operations-webinar-recap-header@2x.png) <br>
# 

# For the last few sessions we have talked about simple linear regression ... <br>
# 
# ![](https://biol609.github.io/lectures/images/03/simple_regression.jpeg) <br>
# 
# We discussed ...
# - __The theory and implementation of simple linear regression in Python__<br>
# - __OLS and MLE methods for estimation of slope and intercept coefficients__  <br>    
# - __Errors (Noise, Variance, Bias) and their impacts on model's performance__ <br>
# - __Confidence and prediction intervals__
# - __And Multiple Linear Regressions__
#     
#  <br> ![](https://memegenerator.net/img/instances/73408711.jpg) <br>
# 
# - __What if we want to predict a discrete variable?__
# 
#     The general idea behind our efforts was to use a set of observed events (samples) to capture the relationship between one or more predictor (AKA input, indipendent) variables and an output (AKA response, dependent) variable. The nature of the dependent variables differentiates *__regression__* and *__classification__* problems. 
#  <br>   ![](https://static.javatpoint.com/tutorial/machine-learning/images/regression-vs-classification-in-machine-learning.png) <br>
# 
#     
#     Regression problems have continuous and usually unbounded outputs. An example is when you’re estimating the salary as a function of experience and education level. Or all the examples we have covered so far! 
#     
#     On the other hand, classification problems have discrete and finite outputs called classes or categories. For example, predicting if an employee is going to be promoted or not (true or false) is a classification problem. There are two main types of classification problems:
# 
#     - Binary or binomial classification: 
#     
#     exactly two classes to choose between (usually 0 and 1, true and false, or positive and negative)
#     
#     - Multiclass or multinomial classification:
#     
#     three or more classes of the outputs to choose from
# 
# 
# - __When Do We Need Classification?__
#     
#     We can apply classification in many fields of science and technology. For example, text classification algorithms are used to separate legitimate and spam emails, as well as positive and negative comments. Other examples involve medical applications, biological classification, credit scoring, and more.
#     
# ## Logistic Regression
# 
# - __What is logistic regression?__
#     Logistic regression is a fundamental classification technique. It belongs to the group of linear classifiers and is somewhat similar to polynomial and linear regression. Logistic regression is fast and relatively uncomplicated, and it’s convenient for users to interpret the results. Although it’s essentially a method for binary classification, it can also be applied to multiclass problems. 
#     
# <br>    ![](https://www.biolegend.com/Files/Images/BioLegend/blog/122118correlationblog/LinearRegresssion.jpg) <br> 
#     
# 
# 

# Logistic regression is a statistical method for predicting binary classes. The outcome or target variable is dichotomous in nature. Dichotomous means there are only two possible classes. For example, it can be used for cancer detection problems. It computes the probability of an event occurrence. Logistic regression can be considered a special case of linear regression where the target variable is categorical in nature. It uses a log of odds as the dependent variable. Logistic Regression predicts the probability of occurrence of a binary event utilizing a logit function. HOW?
# Remember the general format of the multiple linear regression model: 
# <br> ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281880/image1_ga8gze.png) <br>
# 
# Where, y is dependent variable and x1, x2 ... and Xn are explanatory variables. This was, as you know by now, a linear function. There is another famous function known as the *__Sigmoid Function__*, also called *__logistic function__*. Here is the equation for the Sigmoid function: 
# <br> ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281880/image2_kwxquj.png) <br>
# 
# This image shows the sigmoid function (or S-shaped curve) of some variable 𝑥: 
# <br> ![](https://files.realpython.com/media/log-reg-1.e32deaa7cbac.png) <br>
# As you see, The sigmoid function has values very close to either 0 or 1 across most of its domain. It can take any real-valued number and map it into a value between 0 and 1. If the curve goes to positive infinity, y predicted will become 1, and if the curve goes to negative infinity, y predicted will become 0. This fact makes it suitable for application in classification methods since we are dealing with two discrete classes (labels, categories, ...). If the output of the sigmoid function is more than 0.5, we can classify the outcome as 1 or YES, and if it is less than 0.5, we can classify it as 0 or NO. This cutoff value (threshold) is not always fixed at 0.5. If we apply the Sigmoid function on linear regression: 
# <br>![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281880/image3_qldafx.png) <br>
# 
# Notice the difference between linear regression and logistic regression: 
# <br>![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281070/linear_vs_logistic_regression_edxw03.png) <br>
# 
# logistic regression is estimated using Maximum Likelihood Estimation (MLE) approach. Maximizing the likelihood function determines the parameters that are most likely to produce the observed data. 
#     
# Let's work on an example in Python! <br>
# 
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQS4dpT6isEOjJZ2WAahxwOHvpAwYq6Khy4TQ&usqp=CAU) <br>

# ### Diagnosing Diabetes <br>
# 
# ![](https://res.cloudinary.com/grohealth/image/upload/c_fill,f_auto,fl_lossy,h_650,q_auto,w_1085/v1581695681/DCUK/Content/causes-of-diabetes.png) <br>
# 
# 
# 
# The "diabetes.csv" dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. 
# *Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.*
# 
# The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on. 
# 
# |Columns|Info.|
# |---:|---:|
# |Pregnancies |Number of times pregnant|
# |Glucose |Plasma glucose concentration a 2 hours in an oral glucose tolerance test|
# |BloodPressure |Diastolic blood pressure (mm Hg)|
# |SkinThickness |Triceps skin fold thickness (mm)|
# |Insulin |2-Hour serum insulin (mu U/ml)|
# |BMI |Body mass index (weight in kg/(height in m)^2)|
# |Diabetes pedigree |Diabetes pedigree function| 
# |Age |Age (years)|
# |Outcome |Class variable (0 or 1) 268 of 768 are 1, the others are 0|
# 
# 
# Let's see if we can build a logistic regression model to accurately predict whether or not the patients in the dataset have diabetes or not?
# *Acknowledgements:
# Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.*

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Import the dataset:
data = pd.read_csv("diabetes.csv")
data.rename(columns = {'Pregnancies':'pregnant', 'Glucose':'glucose','BloodPressure':'bp','SkinThickness':'skin',
                       'Insulin	':'Insulin','BMI':'bmi','DiabetesPedigreeFunction':'pedigree','Age':'age', 
                              'Outcome':'label'}, inplace = True) 
data.head()


# In[3]:


data.describe()


# In[4]:


#Check some histograms
sns.distplot(data['pregnant'], kde = True, rug= True, color ='orange') 


# In[5]:


sns.distplot(data['glucose'], kde = True, rug= True, color ='darkblue') 


# In[6]:


sns.distplot(data['label'], kde = False, rug= True, color ='purple', bins=2) 


# In[7]:


sns.jointplot(x ='glucose', y ='label', data = data, kind ='kde')


# #### Selecting Features: 
# Here, we need to divide the given columns into two types of variables dependent(or target variable) and independent variable(or feature variables or predictors).

# In[8]:


#split dataset in features and target variable
feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'Insulin', 'bmi', 'pedigree', 'age']
X = data[feature_cols] # Features
y = data.label # Target variable


# #### Splitting Data: 
# To understand model performance, dividing the dataset into a training set and a test set is a good strategy. Let's split dataset by using function train_test_split(). You need to pass 3 parameters: features, target, and test_set size. Additionally, you can use random_state to select records randomly. Here, the Dataset is broken into two parts in a ratio of 75:25. It means 75% data will be used for model training and 25% for model testing:

# In[9]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# #### Model Development and Prediction: 
# First, import the Logistic Regression module and create a Logistic Regression classifier object using LogisticRegression() function. Then, fit your model on the train set using fit() and perform prediction on the test set using predict().

# In[10]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
#logreg = LogisticRegression()
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)


# ![](https://miro.medium.com/max/1200/1*PM4dqcAe6N7kWRpXKwgWag.png) <br>
# - __How to assess the performance of logistic regression?__
# 
#     Binary classification has four possible types of results:
# 
#     - True negatives: correctly predicted negatives (zeros)
#     - True positives: correctly predicted positives (ones)
#     - False negatives: incorrectly predicted negatives (zeros)
#     - False positives: incorrectly predicted positives (ones)
#     
#    We usually evaluate the performance of a classifier by comparing the actual and predicted outputsand counting the correct and incorrect predictions. A confusion matrix is a table that is used to evaluate the performance of a classification model. 
#     
#     <br> ![](https://image.jimcdn.com/app/cms/image/transf/dimension=699x10000:format=png/path/s8ff3310143614e07/image/iab2d53abc26a2bc7/version/1549760945/image.png) <br>
# 
#     Some indicators of binary classifiers include the following:
# 
#     - The most straightforward indicator of classification accuracy is the ratio of the number of correct predictions to the total number of predictions (or observations). 
#     - The positive predictive value is the ratio of the number of true positives to the sum of the numbers of true and false positives.
#     - The negative predictive value is the ratio of the number of true negatives to the sum of the numbers of true and false negatives.
#     - The sensitivity (also known as recall or true positive rate) is the ratio of the number of true positives to the number of actual positives.
#     - The precision score quantifies the ability of a classifier to not label a negative example as positive. The precision score can be interpreted as the probability that a positive prediction made by the classifier is positive.
#     - The specificity (or true negative rate) is the ratio of the number of true negatives to the number of actual negatives. 
# <br>    ![](https://miro.medium.com/max/936/0*R7idSv1bja3CLC8s.png) <br>
#     
# The extent of importance of recall and precision depends on the problem. Achieving a high recall is more important than getting a high precision in cases like when we would like to detect as many heart patients as possible. For some other models, like classifying whether a bank customer is a loan defaulter or not, it is desirable to have a high precision since the bank wouldn’t want to lose customers who were denied a loan based on the model’s prediction that they would be defaulters. 
# There are also a lot of situations where both precision and recall are equally important. Then we would aim for not only a high recall but a high precision as well. In such cases, we use something called F1-score. F1-score is the Harmonic mean of the Precision and Recall: 
# <br> ![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/f1score-300x73.png) <br>
# This is easier to work with since now, instead of balancing precision and recall, we can just aim for a good F1-score and that would be indicative of a good Precision and a good Recall value as well.
# <br>    ![](https://memegenerator.net/img/instances/85090403.jpg) <br>

# #### Model Evaluation using Confusion Matrix: 
# A confusion matrix is a table that is used to evaluate the performance of a classification model. You can also visualize the performance of an algorithm. The fundamental of a confusion matrix is the number of correct and incorrect predictions are summed up class-wise.

# In[11]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred, y_test)
cnf_matrix


# Here, you can see the confusion matrix in the form of the array object. The dimension of this matrix is 2*2 because this model is binary classification. You have two classes 0 and 1. Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions. In the output, 119 and 36 are actual predictions, and 26 and 11 are incorrect predictions.
# 
# #### Visualizing Confusion Matrix using Heatmap: 
# Let's visualize the results of the model in the form of a confusion matrix using matplotlib and seaborn.

# In[12]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Predicted label')
plt.xlabel('Actual label')


# #### Confusion Matrix Evaluation Metrics: 
# 
# Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall.

# In[13]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))


# In[14]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# ![](https://memegenerator.net/img/instances/85090569.jpg)

# ___
# ### Credit Card Fraud Detection <br>
# 
# ![](https://i.pinimg.com/originals/5e/2e/a9/5e2ea94eb6d47c16ece524873234d199.png) <br>
# 
# 
# 
# For many companies, losses involving transaction fraud amount to more than 10% of their total expenses. The concern with these massive losses leads companies to constantly seek new solutions to prevent, detect and eliminate fraud. Machine Learning is one of the most promising technological weapons to combat financial fraud. The objective of this project is to create a simple Logistic Regression model capable of detecting fraud in credit card operations, thus seeking to minimize the risk and loss of the business.
# 
# The dataset used contains transactions carried out by European credit card holders that took place over two days in September 2013, and is a shorter version of a dataset that is available on kaggle at https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3.
# 
# >"It contains only numerical input variables which are the result of a PCA (Principal Component Analysis) transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise."
# 
# 
# |Columns|Info.|
# |---:|---:|
# |Time |Number of seconds elapsed between this transaction and the first transaction in the dataset|
# |V1-V28 |Result of a PCA Dimensionality reduction to protect user identities and sensitive features(v1-v28)|
# |Amount |Transaction amount|
# |Class |1 for fraudulent transactions, 0 otherwise|
# 
# 
# :::{note}
# Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.
# :::
# 
# <hr>
# 
# *__Acknowledgements__*
# The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.
# More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project
# 
# Please cite the following works:
# 
# *Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015*
# 
# *Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon*
# 
# *Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE*
# 
# *Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)*
# 
# *Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier*
# 
# *Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing*
# 
# *Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019*
# 
# *Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019*

# As you know by now, the first step is to load some necessary libraries:

# In[15]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Then, we should read the dataset and explore it using tools such as descriptive statistics:

# In[16]:


# Import the dataset:
data = pd.read_csv("creditcard_m.csv")
data.head()


# As expected, the dataset has 31 columns and the target variable is located in the last one. Let's check and see whether we have any missing values in the dataset:

# In[17]:


data.isnull().sum()


# Great! No missing values!

# In[18]:


data.describe()


# In[19]:


print ('Not Fraud % ',round(data['Class'].value_counts()[0]/len(data)*100,2))
print ()
print (round(data.Amount[data.Class == 0].describe(),2))
print ()
print ()
print ('Fraud %    ',round(data['Class'].value_counts()[1]/len(data)*100,2))
print ()
print (round(data.Amount[data.Class == 1].describe(),2))


# We have a total of 140000 samples in this dataset. The PCA components (V1-V28) look as if they have similar spreads and rather small mean values in comparison to another predictors such as 'Time'. The majority (75%) of transactions are below 81 euros with some considerably high outliers (the max is 19656.53 euros). Around 0.19% of all the observed transactions were found to be fraudulent which means that we are dealing with an extremely unbalanced dataset. An important characteristic of such problems. Although the share may seem small, each fraud transaction can represent a very significant expense, which together can represent billions of dollars of lost revenue each year.
# 
# The next step is to define our **predictors** and **target**:

# In[20]:


#split dataset in features and target variable
y = data.Class # Target variable
X = data.loc[:, data.columns != "Class"] # Features


# The next step would be to split our dataset and define the training and testing sets. The random seed (np.random.seed) is used to ensure that the same data is used for all runs. Let's do a 70/30 split:

# In[21]:


# split X and y into training and testing sets
np.random.seed(123)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)


# Now it is time for model development and prediction! 
# 
# `import` the Logistic Regression module and create a Logistic Regression classifier object using LogisticRegression() function. Then, fit your model on the train set using fit() and perform prediction on the test set using predict().

# In[22]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
#logreg = LogisticRegression()
logreg = LogisticRegression(solver='lbfgs',max_iter=10000)
# fit the model with data  -TRAIN the model
logreg.fit(X_train,y_train)


# In[23]:


# TEST the model
y_pred=logreg.predict(X_test)


# Once the model and the predictions are ready, we can assess the performance of our classifier. First, we need to get our confusion matrix:
# 
# >A confusion matrix is a table that is used to evaluate the performance of a classification model. You can also visualize the performance of an algorithm. The fundamental of a confusion matrix is the number of correct and incorrect predictions are summed up class-wise.

# In[24]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred, y_test)
print(cnf_matrix)
tpos = cnf_matrix[0][0]
fneg = cnf_matrix[1][1]
fpos = cnf_matrix[0][1]
tneg = cnf_matrix[1][0]
print("True Positive Cases are",tpos) #How many non-fraud cases were identified as non-fraud cases - GOOD
print("True Negative Cases are",tneg) #How many Fraud cases were identified as Fraud cases - GOOD
print("False Positive Cases are",fpos) #How many Fraud cases were identified as non-fraud cases - BAD | (type 1 error)
print("False Negative Cases are",fneg) #How many non-fraud cases were identified as Fraud cases - BAD | (type 2 error)


# In[25]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Predicted label')
plt.xlabel('Actual label')


# We should go further and evaluate the model using model evaluation metrics such as accuracy, precision, and recall. These are calculated based on the confustion matrix:

# In[26]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# That is a fantastic accuracy score, isn't it?

# In[27]:


print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))


# In[28]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# Although the accuracy is excellent, the model struggles with fraud detection and has not captured about 30 out of 71 fraudulent transactions.
# 
# Accuracy in a highly unbalanced data set does not represent a correct value for the efficiency of a model. That's where precision, recall and more specifically F1-score as their combinations becomes important:
# 
# - *Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial*
# 
# - *Accuracy can be used when the class distribution is similar while F1-score is a better metric when there are imbalanced classes as in the above case.*
# 
# - *In most real-life classification problems, imbalanced class distribution exists and thus F1-score is a better metric to evaluate our model on.*

# ![](https://media2.giphy.com/media/5nj4ZZWl6QwneEaBX4/source.gif) <br>
# 
# *This notebook was inspired by several blogposts including:* 
# 
# - __"Logistic Regression in Python"__ by __Mirko Stojiljković__ available at* https://realpython.com/logistic-regression-python/ <br>
# - __"Understanding Logistic Regression in Python"__ by __Avinash Navlani__ available at* https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python <br>
# - __"Understanding Logistic Regression with Python: Practical Guide 1"__ by __Mayank Tripathi__ available at* https://datascience.foundation/sciencewhitepaper/understanding-logistic-regression-with-python-practical-guide-1 <br>
# - __"Understanding Data Science Classification Metrics in Scikit-Learn in Python"__ by __Andrew Long__  available at* https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019 <br>
# 
# 
# *Here are some great reads on these topics:* 
# - __"Example of Logistic Regression in Python"__ available at* https://datatofish.com/logistic-regression-python/ <br>
# - __"Building A Logistic Regression in Python, Step by Step"__ by __Susan Li__ available at* https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8 <br>
# - __"How To Perform Logistic Regression In Python?"__ by __Mohammad Waseem__ available at* https://www.edureka.co/blog/logistic-regression-in-python/ <br>
# - __"Logistic Regression in Python Using Scikit-learn"__ by __Dhiraj K__ available at* https://heartbeat.fritz.ai/logistic-regression-in-python-using-scikit-learn-d34e882eebb1 <br>
# - __"ML | Logistic Regression using Python"__ available at* https://www.geeksforgeeks.org/ml-logistic-regression-using-python/ <br>
# 
# *Here are some great videos on these topics:* 
# - __"StatQuest: Logistic Regression"__ by __StatQuest with Josh Starmer__ available at* https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe <br>
# - __"Linear Regression vs Logistic Regression | Data Science Training | Edureka"__ by __edureka!__ available at* https://www.youtube.com/watch?v=OCwZyYH14uw <br>
# - __"Logistic Regression in Python | Logistic Regression Example | Machine Learning Algorithms | Edureka"__ by __edureka!__ available at* https://www.youtube.com/watch?v=VCJdg7YBbAQ <br>
# - __"How to evaluate a classifier in scikit-learn"__ by __Data School__ available at* https://www.youtube.com/watch?v=85dtiMz9tSo <br>
# - __"How to evaluate a classifier in scikit-learn"__ by __Data School__ available at* https://www.youtube.com/watch?v=85dtiMz9tSo <br>

# ___
# ![](https://media2.giphy.com/media/dNgK7Ws7y176U/200.gif) <br>
# 

# ## Exercise: Logistic Regression in Engineering <br>
# 
# ### Think of a few applications of Logistic Regression in Engineering?
# 
# #### _Make sure to cite any resources that you may use._ 

# ## topic
# From Wikipedia (https://en.wikipedia.org/wiki/Logistic_regression) (emphasis is mine):
# 
# In statistics, the **logistic model** (or logit model) is used to model the probability of a certain **class** or **event** existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc. Each object being detected in the image would be assigned a probability between 0 and 1, with a sum of one.
# 
# Logistic regression is a **statistical model** that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression). Mathematically, a **binary logistic model** has a dependent variable with two possible values, such as pass/fail which is represented by an indicator variable, where the two values are labeled "0" and "1". In the **logistic model**, the log-odds (the logarithm of the odds) for the value labeled "1" is a linear combination of one or more independent variables ("predictors"); the independent variables can each be a binary variable (two classes, coded by an indicator variable) or a continuous variable (any real value). 
# 
# The corresponding probability of the value labeled "1" can vary between 0 (certainly the value "0") and 1 (certainly the value "1"), hence the labeling; the function that converts log-odds to probability is the logistic function, hence the name. The unit of measurement for the log-odds scale is called a logit, from logistic unit, hence the alternative names. Analogous models with a different sigmoid function instead of the logistic function can also be used, such as the probit model; the defining characteristic of the logistic model is that increasing one of the independent variables multiplicatively scales the odds of the given outcome at a constant rate, with each independent variable having its own parameter; for a binary dependent variable this generalizes the odds ratio.
# 
# In a binary logistic regression model, the dependent variable has two levels (categorical). Outputs with more than two values are modeled by multinomial logistic regression and, if the multiple categories are ordered, by ordinal logistic regression (for example the proportional odds ordinal logistic model). The logistic regression model itself simply **models probability of output** in terms of input and does not perform statistical classification (it is not a classifier), though it can be used to make a classifier, for instance by choosing a cutoff value and classifying inputs with probability greater than the cutoff as one class, below the cutoff as the other; this is a common way to make a binary classifier. The coefficients are generally not computed by a closed-form expression, unlike linear least squares; see § Model fitting. $\dots$
# 
# Now lets visit the Wiki and learn more https://en.wikipedia.org/wiki/Logistic_regression

# Now lets visit our friends at towardsdatascience https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc  Here we will literally CCMR the scripts there.

# We need the data,  a little searching and its here https://gist.github.com/curran/a08a1080b88344b0c8a7 after download and extract we will need to rename the database

# lorem ipsum

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[30]:


df = pd.read_csv('iris-data.csv')


# In[31]:


df.head()


# In[32]:


df.describe()


# In[33]:


df.info()


# In[34]:


#Removing all null values row
df = df.dropna(subset=['petal_width_cm'])
df.info()


# In[35]:


#Plot


# In[36]:


sns.pairplot(df, hue='class', height=2.5)


# From the plots it can be observed that there is some abnormality in the class name. Let's explore further

# In[37]:


df['class'].value_counts()


# Two observations can be made from the above results
# - For 5 data points 'Iris-versicolor' has been specified as 'versicolor' 
# - For 1 data points, 'Iris-setosa' has been specified as 'Iris-setossa'

# In[38]:


df['class'].replace(["Iris-setossa","versicolor"], ["Iris-setosa","Iris-versicolor"], inplace=True)
df['class'].value_counts()


# # Simple Logistic Regression

# Consider only two class 'Iris-setosa' and 'Iris-versicolor'. Dropping all other class

# In[39]:


final_df = df[df['class'] != 'Iris-virginica']


# In[40]:


final_df.head()


# # Outlier Check

# In[41]:


sns.pairplot(final_df, hue='class', height=2.5)


# From the above plot, sepal_width and sepal_length seems to have outliers. To confirm let's plot them seperately

# SEPAL LENGTH

# In[42]:


final_df.hist(column = 'sepal_length_cm',bins=20, figsize=(10,5))


# It can be observed from the plot, that for 5 data points values are below 1 and they seem to be outliers. So, these data points
# are considered to be in 'm' and are converted to 'cm'.

# In[43]:


final_df.loc[final_df.sepal_length_cm < 1, ['sepal_length_cm']] = final_df['sepal_length_cm']*100
final_df.hist(column = 'sepal_length_cm',bins=20, figsize=(10,5))


# SEPAL WIDTH

# In[44]:


final_df = final_df.drop(final_df[(final_df['class'] == "Iris-setosa") & (final_df['sepal_width_cm'] < 2.5)].index)


# In[45]:


sns.pairplot(final_df, hue='class', height=2.5)


# Successfully removed outliers!!

# # Label Encoding

# In[46]:


final_df['class'].replace(["Iris-setosa","Iris-versicolor"], [1,0], inplace=True)


# In[47]:


final_df.head()


# # Model Construction

# In[48]:


inp_df = final_df.drop(final_df.columns[[4]], axis=1)
out_df = final_df.drop(final_df.columns[[0,1,2,3]], axis=1)
#
scaler = StandardScaler()
inp_df = scaler.fit_transform(inp_df)
#
X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)


# In[49]:


X_tr_arr = X_train
X_ts_arr = X_test
#y_tr_arr = y_train.as_matrix() # method deprecated several versions ago
#y_ts_arr = y_test.as_matrix()
y_tr_arr = y_train.values
y_ts_arr = y_test.values


# In[50]:


print('Input Shape', (X_tr_arr.shape))
print('Output Shape', X_test.shape)


# In[51]:


def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b


# In[52]:


def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result


# In[53]:


def model_optimize(w, b, X, Y):
    m = X.shape[0]
    
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost
    


# In[54]:


def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w,b,X,Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        #weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #
        
        if (i % 100 == 0):
            costs.append(cost)
            #print("Cost after %i iteration is %f" %(i, cost))
    
    #final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}
    
    return coeff, gradient, costs


# In[55]:


def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred


# In[56]:


#Get number of features
n_features = X_tr_arr.shape[1]
print('Number of Features', n_features)
w, b = weightInitialization(n_features)
#Gradient Descent
coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001,no_iterations=4500)
#Final prediction
w = coeff["w"]
b = coeff["b"]
print('Optimized weights', w)
print('Optimized intercept',b)
#
final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)
#
m_tr =  X_tr_arr.shape[0]
m_ts =  X_ts_arr.shape[0]
#
y_tr_pred = predict(final_train_pred, m_tr)
print('Training Accuracy',accuracy_score(y_tr_pred.T, y_tr_arr))
#
y_ts_pred = predict(final_test_pred, m_ts)
print('Test Accuracy',accuracy_score(y_ts_pred.T, y_ts_arr))


# In[57]:


plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Cost reduction over time')
plt.show()


# In[58]:


from sklearn.linear_model import LogisticRegression


# In[59]:


clf = LogisticRegression()


# In[60]:


clf.fit(X_tr_arr, y_tr_arr)


# In[61]:


print (clf.intercept_, clf.coef_)


# In[62]:


pred = clf.predict(X_ts_arr)


# In[63]:


print ('Accuracy from sk-learn: {0}'.format(clf.score(X_ts_arr, y_ts_arr)))


# In[ ]:





# ## topic
# 
# ### Subtopic
# lorem ipsum
# 
# 

# ## References
# 1. Chan, Jamie. Machine Learning With Python For Beginners: A Step-By-Step Guide with Hands-On Projects (Learn Coding Fast with Hands-On Project Book 7) (p. 2). Kindle Edition. 

# In[ ]:




