#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Workflow
# 
# Despite the diverse applications of machine learning, most machine learning projects follow a typical workflow.  Prior to examination of workflow consider ordinary problem solving.
# 
# :::{admonition} Computational Thinking and Data Science
# 
# Computational thinking (CT) refers to the thought processes involved in expressing solutions as computational steps or algorithms that can be carried out by a computer. 
# 
# CT is literally a process for breaking down a problem into smaller parts, looking for patterns in the problems, identifying what kind of information is needed, developing a step-by-step solution, and implementing that solution. 
# 
# 1. **Decomposition** is the process of taking a complex problem and breaking it into more manageable sub-problems. Decomposition often leaves a framework of sub-problems that later have to be assembled (system integration) to produce a desired solution.
# 2. **Pattern Recognition** refers to finding similarities, or shared characteristics of problems, which allows a complex problem to become easier to solve, and allows use of same solution method for each occurrence of the pattern. 
# 3. **Abstraction** is the process of identifying important characteristics of the problem and ignore characteristics that are not important. We use these characteristics to create a representation of what we are trying to solve.
# 4. **Algorithms** are step-by-step instructions of how to solve a problem 
# 5. **System Integration** (implementation)is the assembly of the parts above into the complete (integrated) solution.  Integration combines parts into a program which is the realization of an algorithm using a syntax that the computer can understand. 
# 
# :::
# 
# ## Problem Solving Protocol
# 
# Many engineering courses emphasize a problem solving process that somewhat parallels the [scientific method](https://en.wikipedia.org/wiki/Scientific_method) as one example of an effective problem solving strategy. Stated as a protocol it goes something like:
# 
# 1. Observation: Formulation of a question
# 2. Hypothesis: A  conjecture that may explain observed behavior. Falsifiable by an experiment whose outcome conflicts with predictions deduced from the hypothesis
# 3. Prediction: How the experiment should conclude if hypothesis is correct 
# 4. Testing: Experimental design, and conduct of the experiment.
# 5. Analysis: Interpretation of experimental results
#  
# This protocol can be directly adapted to computational problems as:
# 
# 1. Define the problem (problem statement)
# 2. Gather information (identify known and unknown values, and governing equations)
# 3. Generate and evaluate potential solutions
# 4. Refine and implement a solution
# 5. Verify and test the solution.
# 
# For actual computational methods the protocol becomes:
# 
# 1. Explicitly state the problem
# 2. State:
#   - Input information
#   - Governing equations or principles, and 
#   - The required output information.
# 3. Work a sample problem by-hand for testing the general solution.
# 4. Develop a general solution method (coding).
# 5. Test the general solution against the by-hand example, then apply to the real problem.
# 
# Oddly enough the first step is the most important and sometimes the most difficult. In a practical problem, step 2 is sometimes difficult because a skilled programmer is needed to translate the governing principles into an algorithm for the general solution (step 4).
# 
# We can compare the steps above to a visual representation of the process below from [Machine Learning Techniques for Civil Engineering Problems](http://ce-5319-webroot/3-Readings/MachineLearningTechniquesforCivilEngineeringProblems.pdf).
# 
# <figure align="center">
# <img src="http://54.243.252.9/ce-5319-webroot/ce5319jb/lessons/lesson2/ProblemSolvingProtocol.png" width="800"> <figcaption>Figure 2.1. Problem Solving Protocol Diagram </figcaption>
# </figure>
# 

# ## ML Workflow Steps
# 
# A typical machine learning workflow  includes some (or all) of the steps listed below adapted from [Machine Learning With Python For Beginners](https://www.amazon.com/Machine-Learning-Python-Beginners-Hands/dp/B09BT7YCCM).  As you examine the steps compare them to the more classical problem solving approaches to recognize the parallels.
# 
# - Step 1. Identify, Collect, and Loading Data. (Data Wrangling) The first step to any machine learning project is to load the data. Based on the data at hand, we would need different libraries to load the respective data.  If we are using a python derivative to perform the modeling then for loading CSV files, we need the pandas library. For loading 2D images, we can use the Pillow or OpenCV library. 
# 
# - Step 2. Examine the data. (Exploratory Data Analysis) Assuming the data has been loaded correctly, the next step is to examine the data to get a general feel for the dataset. Let us take the case of a simple CSV file-based dataset. For starters, we can look at the dataset’s shape (i.e., the number of rows and columns in the dataset). We can also peek inside the dataset by looking at its first 10 or 20 rows. In addition, we can perform fundamental analysis on the data to generate some descriptive statistical measures (such as the mean, standard deviation, minimum and maximum values). Last but not least, we can check if the dataset contains missing data. If there are missing values, we need to handle them.
# 
# - Step 3. Split the Dataset. (Training and Testing Subsets) Before we handle missing values or do any form of computation on our dataset, we typically typically split it into training and test subsets. A common practice is to use 80% of the dataset for training and 20% for testing although the proportions are up to the program designer. The training subset is the actual dataset used for training the model. After the training process is complete, we can use the test subset to evaluate how well the model generalizes to unseen data (i.e., data not used to train the model). It is crucial to treat the test subset as if it does not exist during the training stage. Therefore, we should not touch it until we have finished training our model and are ready to evaluate the selected model.
# 
# :::{note}
# A huge philosophical issue arises if the testing set suggests that we have a crappy model - at that juncture if we change the model at all the testing set has just been used for training and calls into question the whole split data process.  This dilemma is rarely discussed in the literature, but is an important ethical issue to keep in mind when letting a machine control things that can kill. Most people think hellfire missles and drones, but water treatment plants under unattended autonomous control can kill as effectively as a missle.
# :::
# 
# - Step 4. Data Visualization (Exploratory Data Analysis) After splitting the dataset, we can plot some graphs to better understand the data we are investigating. For instance, we can plot scatter plots to investigate the relationships between the features (explainatory variables, predictor variables, etc.) and the target (response) variable.
# 
# - Step 5. Data Preprocessing. (Data Structuring)   The next step is to do data preprocessing. More often than not, the data that we receive is not ready to be used immediately. Some problems with the dataset include missing values, textual and categorical data (e.g., “Red”, “Green”, and “Blue” for color), or range of features that differ too much (such as a feature with a range of 0 to 10,000 and another with a range of 0 to 5). Most machine learning algorithms do not perform well when any of the issues above exist in the dataset. Therefore, we need to process the data before passing it to the algorithm.
# 
# - Step 6. Model Training. After preparing the data, we are ready to train our models. Based on the previous steps of analyzing the dataset, we can select  appropriate machine learning algorithms and build models using those algorithms. 
# 
# - Step 7. Performance Evaluation.  After building the models, we need to evaluate our models using different metrics and select the best-performing model for deployment. At this stage, the technical aspects of a  machine learning project is more or less complete.
# 
# - Step 8. Deploy the Model.  Deploy the best-performing model.  Build a user interface so customers/designers/clients can apply the model to their needs.
# 
# - Step 9.  Monitoring and maintaining the model.  This step is quite overlooked in most documents on ML, even more so than deploy.  This step would audit post-deployment use and from time-to-time re-train the model with newer data.  Some industries perform this retraining almost continuously; in Civil Engineering the retraining may happen on a decade-long time scale perhaps even longer.  It depends on the consequence of failure, and the model value.
# 
# Consider Figure 2.1 below as another representation of workflow from [Machine Learning Workflow Explained](https://towardsdatascience.com/the-machine-learning-workflow-explained-557abf882079).  
# 
# <figure align="center">
# <img src="https://miro.medium.com/max/700/1*XgcF3ayEH2Q8JEbZx8D09Q.png" width="800"> <figcaption>Figure 2.2. ML Workflow Diagram </figcaption>
# </figure>
# 
# The figure components largely address each list item, some are shared, others combined - nevertheless we have a suitable workflow diagram.  The author of the diagram places data at the top and cycles from there, a subtle symbolic placement with which I wholly agree.  Read his blog post to see his thinking in that respect.

# ## References
# 
# 1. Chan, Jamie. Machine Learning With Python For Beginners: A Step-By-Step Guide with Hands-On Projects (Learn Coding Fast with Hands-On Project Book 7) (p. 5-7). Kindle Edition. 
# 
# 2. [Thevapalan, Arunn (2021). The Machine Learning Workflow Explained (and How You Can Practice It Now). Towards Data Science Blog](https://towardsdatascience.com/the-machine-learning-workflow-explained-557abf882079)
# 
# 3. [Computational and Inferential Thinking Ani Adhikari and John DeNero, Computational and Inferential Thinking, The Foundations of Data Science, Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND) Chapter 1](https://www.inferentialthinking.com/chapters/01/what-is-data-science.html)
# 
# 4. [Learn Python the Hard Way (Online Book)](https://learnpythonthehardway.org/book/)  Recommended for beginners who want a complete course in programming with Python.
# 
# 5. [LearnPython.org (Interactive Tutorial)](https://www.learnpython.org/)  Short, interactive tutorial for those who just need a quick way to pick up Python syntax.
# 
# 6. [How to Think Like a Computer Scientist (Interactive Book)](https://runestone.academy/runestone/books/published/thinkcspy/index.html) Interactive "CS 101" course taught in Python that really focuses on the art of problem solving. 
# 
# 7. [How to Learn Python for Data Science, The Self-Starter Way](https://elitedatascience.com/learn-python-for-data-science) 
# 

# In[ ]:



