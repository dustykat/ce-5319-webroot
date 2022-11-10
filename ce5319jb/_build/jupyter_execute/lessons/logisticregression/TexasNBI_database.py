#!/usr/bin/env python
# coding: utf-8

# # Processing Texas Database
# 
# Following the process described in [Prediction of Bridge Component Ratings Using Ordinal Logistic Regression Model](http://54.243.252.9/ce-5319-webroot/3-Readings/bridgeDatabase9797584.pdf) but for Texas data.

# In[1]:


# Playing with Texas Data
local_file_name='2021TexasNBIData.csv'
import pandas as pd
# Read the NBI Database
texasdb = pd.read_csv(local_file_name)


# In[2]:


# Verify the Structure
texasdb.head()


# In[3]:


# Rename The Index
#texasdb = texasdb.rename_axis('Index')
#texasdb.head()


# In[4]:


txculv = texasdb.loc[texasdb['STRUCTURE_TYPE_043B'] == 19]


# In[5]:


list(texasdb)


# In[6]:


txculv = texasdb[['CULVERT_COND_062','YEAR_BUILT_027','ADT_029','DESIGN_LOAD_031','OPERATING_RATING_064','STRUCTURAL_EVAL_067']].copy()


# In[7]:


txculv.head()


# In[8]:


age = 2022 - txculv['YEAR_BUILT_027'] # age of culvert, surrogate for service life


# In[9]:


drat = txculv['OPERATING_RATING_064'].fillna(1) # operating rating (in tons), nan is 1
drat = pd.to_numeric(drat) # coerce to numeric


# In[10]:


steval = txculv['STRUCTURAL_EVAL_067'].fillna(1).replace('*','1') # structural eval, nan is 1 (not in original NBI coding tables)
steval = pd.to_numeric(steval) # coerce to numeric


# In[11]:


adt =  txculv['ADT_029'].fillna(1) # 
adt = pd.to_numeric(adt) # coerce to numeric


# In[12]:


frame = { 'AGE': age, 'DRAT': drat, "ADT" :adt, "STEVAL":steval}
#frame = { 'AGE': age, 'DRAT': drat, "STEVAL":steval}
#Creating DataFrame by passing Dictionary
X = pd.DataFrame(frame)  # our design matrix


# Now build our target
# 
# Prepare the target variable, from the culvert condition. The series has NaN and we convert these to a goofy string

# In[13]:


pre_target = txculv['CULVERT_COND_062'].fillna('11').replace('N','10') # Nan to 11, "N" to 10 so all codes able to be numeric
print("Type ",type(pre_target[0]))


# then convert from string to numeric.

# In[14]:


pre_target = pd.to_numeric(pre_target)
print("Type ",type(pre_target[0]))


# Need a function to select the two conditions

# In[15]:


def isok(value_int): # function to interpret condition rating and issue binary state 0 == fail 1 == OK
    cut = 6
    if value_int > cut:
        isok = 1
    elif value_int <= cut:
        isok = 0
    return(isok)

y = pre_target.apply(isok) # our target vector


# Now check that our design matrix and target vector have correct structure

# In[16]:


X.describe()


# In[17]:


y.describe()


# In[18]:


# Now we can do some machine learning
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[19]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
#logreg = LogisticRegression()
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)


# In[20]:


print(logreg.intercept_[0])
print(logreg.coef_)
#y.head()


# In[21]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred, y_test)
cnf_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
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
plt.xlabel('Actual label');


# In[22]:


yrbld = txculv['YEAR_BUILT_027']
type(yrbld[0])


# In[23]:


#pre_target = txculv['CULVERT_COND_062']


# Structurally Deficient (SD): This term was previously defined in https://www.fhwa.dot.gov/bridge/0650dsup.cfm as having a condition rating of 4 or less for Item 58 (Deck), Item 59 (Superstructure), Item 60 (Substructure), or Item 62 (Culvert), OR having an appraisal rating of 2 or less for Item 67 (Structural Condition) or Item 71 (Waterway Adequacy) Beginning with the 2018 data archive, this term will be defined in accordance with the Pavement and Bridge Condition Performance Measures final rule, published in January of 2017, as a classification given to a bridge which has any component [Item 58, 59, 60, or 62] in Poor or worse condition [code of 4 or less].
# 
# This capacity rating, referred to as the operating rating, will result
# in the absolute maximum permissible load level to which the structure
# may be subjected for the vehicle type used in the rating. Code the
# operating rating as a 3-digit number to represent the total mass in
# metric tons of the entire vehicle measured to the nearest tenth of a
# metric ton (with an assumed decimal point).

# In[24]:


#type(pre_target[0])


# Prepare the target variable, the culvert condition as above (not processing just yet).
# The series has NaN and we convert these to a goofy string
# then convert from string to numeric.

# Now make a proper target

# In[25]:


#target


# In[ ]:




