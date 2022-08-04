#!/usr/bin/env python
# coding: utf-8

# # Classification Engines
# 
# In contrast to prediction engines the other category of great interest in machine learning is classification.
# 
# Computers are mostly calculators; They are very very fast at doing arithmetic. This feature is great for doing tasks that match what a calculator does: summing numbers to work out sales, applying percentages to work out tax, plotting graphs of existing data, solving [tractrix](https://mathworld.wolfram.com/Tractrix.html) equations to create smoking holes in the ocean where there used be an aircraft carrier.  Even watching TV or streaming music through your computer doesn’t involve much more than the computer executing simple arithmetic instructions repeatedly. 
# Reconstructing a video frame from the ones and zeros that are piped across the internet to your computer is done using arithmetic not much more complex than the sums we did in grade school.  
# Adding up numbers really quickly  thousands, or even millions of times a second  may be impressive  but it isn’t intelligence. 
# 
# A human may find it hard to do large sums very quickly but the  process of doing it doesn’t require much intelligence at all (the size of the federal government, and number of elected officials is a testament to this fact). It simply requires an ability to follow  very basic instructions, and this is what the electronics inside a computer does.  Now let’s flips things and turn the tables on computers!  Look at the following images and see if you can recognise what they contain:
# 
# ![](http://54.243.252.9/engr-1330-webroot/1-Lessons/Lesson22/pictures.png)
# 
# You can immediately recognize people, a cat, and a tree -- you are able to classify the pictures very fast.
# We can process the quite large amount of information that the images contain, and very  successfully process it to recognise what’s in the image. This kind of task isn’t easy for computers  in fact it’s incredibly difficult. 
# 
# Consider what happens when we reduce the information into a 27X27 pixel map to see one reason why classification is hard for a machine -- a resolution issue, also we will see how at reduce resolution the pictures look alike.  First some image processing libraries:

# In[1]:


import numpy              # useful numerical routines
import scipy.special      # special functions library
import scipy.misc         # image processing code
import imageio            # image processing library
import matplotlib.pyplot  # import plotting routines


# Here's the files containing the images if you want to try this at home
# 
# [people784.png](http://54.243.252.9/ce-5319-webroot/ce5319jb/lessons/lesson4/people784.png)<br>
# [cat784.png](http://54.243.252.9/ce-5319-webroot/ce5319jb/lessons/lesson4/cat784.png)<br>
# [tree784.png](http://54.243.252.9/ce-5319-webroot/ce5319jb/lessons/lesson4/tree784.png)<br>
# 
# Now we read and render the people image in reduced resolution about 1/2 of the original -- still barely recognizable for us humans.  The image is converted to an array of floating point values from 0 to 255 (256 different values), each representing a different shade of grey.   

# In[2]:


########### suppress warnings ######
import warnings                   ##
warnings.filterwarnings('ignore') ##
####################################
img_array = imageio.imread("people784.png", as_gray = True) # read file ignore rgb, only gray scale
img_data0 = 255.0 - img_array.reshape(784) 
img_data0 = ((img_data0/255.0)*0.99) + 0.01
matplotlib.pyplot.imshow(numpy.asfarray(img_data0).reshape((28,28)),cmap = 'Greys') # construct a graphic object #
matplotlib.pyplot.show() # show the graphic object to a window #
matplotlib.pyplot.close('all')


# Now render the cat image in reduced resolution about 1/2 of the original -- still  recognizable for us humans

# In[3]:


img_array = imageio.imread("cat784.png", as_gray = True)
img_data1 = 255.0 - img_array.reshape(784)
img_data1 = ((img_data1/255.0)*0.99) + 0.01
matplotlib.pyplot.imshow(numpy.asfarray(img_data1).reshape((28,28)),cmap = 'Greys') # construct a graphic object #
matplotlib.pyplot.show() # show the graphic object to a window #
matplotlib.pyplot.close('all')


# Now render the tree image in reduced resolution about 1/3 of the original -- still  recognizable for us humans

# In[4]:


img_array = imageio.imread("tree784.png", as_gray = True)
img_data2 = 255.0 - img_array.reshape(784)
img_data2 = ((img_data2/255.0)*0.99) + 0.01
matplotlib.pyplot.imshow(numpy.asfarray(img_data2).reshape((28,28)),cmap = 'Greys') # construct a graphic object #
matplotlib.pyplot.show() # show the graphic object to a window #
matplotlib.pyplot.close('all')


# In[5]:


print("people784 statistics : ",img_data0.mean(),img_data0.var())
print("cat784 statistics : ",img_data1.mean(),img_data1.var())
print("tree784 statistics : ",img_data2.mean(),img_data2.var())


# Using the image statistics, which is just the gray-scale value of each pixel (0-255), we see that the images are different with this simple metric but not by much
# 
#     Image       Mean           Variance
#     People    0.48325375     0.06275265
#     Cat       0.60355407     0.023282547
#     Tree      0.484061       0.049499817
#     
# If we used just a statistical description, in the mean people and tree are the same, whereas a cat is different. But not all cats will have the same mean (or variance).  So simplistic numerical descriptors are useless, we need more that a couple of metrics for the image perhaps higher moments, or a way to consider all pixels at once -- sort of like a regression model. 
# 
# We humans naturally fill in missing information and can classify very fast  -- cognative scientists think (now thats a pun!) that our mind performs "regressions" on the whole image and reduces it to a set of classifiers then these are compared in our brain to historical results and the classification that throw off the most dopamine (our brain's drug of choice) is selected.  It happens fast because the chemical reactions involved can be processed in parallel, the message is sent evreywhere at once and the molecules themselves don't even have to arrive for the classification to occur.
# 
# Anyway the whole process of taking inputs and determining which category or class it belongs is called classification.

# First render the people image in reduced resolution about 1/2 of the original -- still barely recognizable for us humans

# ## A Simple Classification Machine
# 
# Consider the three images below
# 
# 
# 
# 
# But what's a computer to do?  We can have it convert those images to arrays of numbers and try to see if some statistical measures are different for each image then declare all future images with the same statistics as our one cat to be cats.
# 
# Lets see how that goes

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

# In[ ]:




