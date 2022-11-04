#!/usr/bin/env python
# coding: utf-8

# # Anatomy of a Machine Learning Algorithm
# 
# The fundamental building blocks of a ML algorithm, whether **homebrew** or using a **package** consist of three major components:
# 
# 1. a function to assess the quality of a prediction or classification.  Typically a loss function which we want to minimize, or a likelihood function which we want to maximize.  It goes by many names: objective, cost, loss, merit are common names for this function.
# 2. an objective criterion (a goal) based on the loss function (maximize or minimize), and
# 3. an optimization routine that uses the training data to find a solution to the optimization problem posed by the objective criterion.
# 
# The remainder of this Jupyter Book explores these ideas as well as supporting efforts required for useful ML model building for Civil Engineers (actually anyone).
# 

# ## References
# 
# 1. [Andriy Burkov (2019) The Hundred-Page Machine Learning Book](https://www.dropbox.com/s/xpd5x6p6jte3th5/Chapter4.pdf?dl=0)

# In[ ]:




