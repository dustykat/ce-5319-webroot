#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning
# 
# Reinforcement learning (RL) is a type of ML where the "machine" lives in an "environment" and detects the **state** of the environment (maybe with sensors) as a feature vector, and can generate actions (control functions like in a [PID controller](https://en.wikipedia.org/wiki/PID_controller))) in any state.  Different actions provide rewards and can move the machine into another "environmental" state.  The goal of reinforcement learning is to learn a **policy** (similar to the **model** in supervised learning) that takes the curent feature vector of a state and outputs a "non-inferior" action (**policy**) to apply in that state.  The action is said to be optimal if it maximizes the expected average reward (here *expected* is in the statistical sense that is the integral of the product of reward value and probability of getting that reward).
# 
# RL solves problems where decision making is sequential, and the goal is somewhat long-term.  Logistics (Amazon, FedEx, UPS), resource management (fuel production, ...) , robotics (PUMA, wharehouse robots, autonomous killing machines,...), and game-playing (not fun games, but "wargaming" to guide decision makers) are all general categoires of human activity that leverages RL.
# 
# [Digital twin](https://www.ibm.com/topics/what-is-a-digital-twin) simulation and control most probably falls squarely in this category of ML.
# :::{admonition}From wikipedia ...
# Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.
# 
# Reinforcement learning differs from supervised learning in not needing labelled input/output pairs be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).[1] Partially supervised RL algorithms can combine the advantages of supervised and RL algorithms.
# 
# The environment is typically stated in the form of a Markov decision process (MDP), because many reinforcement learning algorithms for this context use dynamic programming techniques. The main difference between the classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the MDP and they target large MDPs where exact methods become infeasible.
# :::

# ## topic
# 

# lorem ipsum
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
# 
# 2. [Applied Dynamic Programming Bellman and Dreyfus (1962)](http://54.243.252.9/ce-5319-webroot/3-Readings/R352.pdf)
# 
# 3. [Reinforcement Learning (Wikipedia)](https://en.wikipedia.org/wiki/Reinforcement_learning)
# 
# 4. [Digital Twins (IBM)](https://www.ibm.com/topics/what-is-a-digital-twin)

# In[ ]:




