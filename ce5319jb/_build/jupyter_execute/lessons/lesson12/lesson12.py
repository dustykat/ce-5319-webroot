#!/usr/bin/env python
# coding: utf-8

# # Non-Linear Regression

# Nonlinear regression is a form of regression analysis in which observational data are modeled by a function which is a **nonlinear** combination of the model parameters and depends on one or more independent (predictor, explainatory, feature ...) variables. The data are fitted by a method of successive approximations. 
# 

# In nonlinear regression, a statistical model of the form,
# 
# $${\displaystyle \mathbf {y} \sim f(\mathbf {x} ,{\boldsymbol {\beta }})}$$
# 
# relates a vector of independent variables, $ \mathbf {x}$ , and its associated observed dependent variables, $\mathbf {y} $. The function $ f$ is nonlinear in the components of the vector of parameters $ \beta$ , but otherwise arbitrary. For example, the Michaelis–Menten model for enzyme kinetics has two parameters and one independent variable, related by $ f $ by
# 
# $$f(x,{\boldsymbol \beta })={\frac {\beta _{1}x}{\beta _{2}+x}}$$
# 
# :::{note}
# In this example from Wikipedia, domain knowledge is implicit in the machine learning aspect of the example.  As said in class, we have to understand the process we are trying to model to some level, otherwise a mere statistician could ahndle the work (and most can with our guidance!)
# :::
# 
# This function is nonlinear because it cannot be expressed as a linear combination of the two $ \beta $s.
# 
# Systematic error may be present in the independent variables but its treatment is outside the scope of regression analysis. If the independent variables are not error-free, this is an errors-in-variables model, also outside this scope.
# 
# Other examples of nonlinear functions include exponential functions, logarithmic functions, trigonometric functions, power functions, Gaussian function, and Lorentz distributions. Some functions, such as the exponential or logarithmic functions, can be transformed so that they are linear. When so transformed, standard linear regression can be performed but must be interpreted with caution. 
# 
# In general, there is no closed-form expression for the best-fitting parameters, as there is in linear regression. Usually numerical **optimization algorithms** are applied to determine the best-fitting parameters. Again in contrast to linear regression, there may be many local minima of the function to be optimized and even the global minimum may produce a biased estimate. In practice, estimated values of the parameters are used, in conjunction with the optimization algorithm, to attempt to find the global minimum of a sum of squares or some other objective (merit, cost, value, quality) function.
# 

# lorem ipsum

# ## topic
# 
# ### Subtopic
# lorem ipsum
# 
# 

# ## References
# 1. Chan, Jamie. Machine Learning With Python For Beginners: A Step-By-Step Guide with Hands-On Projects (Learn Coding Fast with Hands-On Project Book 7) (p. 2). Kindle Edition. 

# In[ ]:




