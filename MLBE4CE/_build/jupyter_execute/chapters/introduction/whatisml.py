#!/usr/bin/env python
# coding: utf-8

# # What is Machine Learning?
# 
# Machine learning is a terminology for computer programs that provide computers the ability to perform a task without being explicitly programmed for that task.
# 
# As an example, suppose we want to sort emails into promotional and non-promotional emails. In conventional programming, we can do this using a set of hard-coded rules or conditional statements. For instance, one possible rule is to classify an email as promotional if it contains the words “Discount”, “Sale”, or “Free Gift”. We can also classify an email as non-promotional if the email address includes “.gov” or “.edu”.  The problem with such an approach is that it is challenging to come up with the rules. 
# 
# For instance, while most emails from addresses that contain “.edu” are likely to be non-promotional (such as an email from your thesis supervisor), it is also possible for educational institutions to send promotional emails advertising their courses. It is almost impossible to come up with a set of rules that considers all possible scenarios.
# 
# Machine learning can improve the sorting program by identifying each email’s unique attributes and autonomously synthesize rules to automate the sorting process, thereby preventing the need for manually engineered rules.
# 
# For a machine to do that, we provide it with data. The goal is for the machine to infer useful rules directly from the data, using what are known as machine learning algorithms. In a nutshell, machine learning algorithms are made up of formulas and procedures derived from mathematical concepts in linear algebra, calculus, probability, statistics, and other fields. These formulas and procedures are implemented in programming code and used to perform calculations on our data. After performing the calculations, the algorithm typically generates an output known as a model (data model, prediction engine model, classification model - there are a lot of equivalent names). The process of generating a model is known as training. The model describes the rules, numbers, and any other algorithm-specific data structures that our machine learned from the data. Our machine can then use the model to perform the rules to new data.

# ## A Classification Engine
# 
# You’ve just arrived on some small pacific island
# 
# <figure align="center">
# <img src="http://54.243.252.9/ce-5319-webroot/ce5319jb/lessons/lesson2/island.png" width="400"> <figcaption>Figure 2.1. Small Pacific Island </figcaption>
# </figure>
# 
# You soon find out that papayas are a substantial ingredient in the local diet
# 
# <figure align="center">
# <img src="http://54.243.252.9/ce-5319-webroot/ce5319jb/lessons/lesson2/papaya.png" width="400"> <figcaption>Figure 2.2. Papaya Tree </figcaption>
# </figure>
# 
# Your obvious question: are papayas tasty?
# 
# From your **prior** experience you recall that softness, and color are good indicators of tastiness, the goal is to generalize this experience into a classification rule
# 
# <figure align="center">
# <img src="http://54.243.252.9/ce-5319-webroot/ce5319jb/lessons/lesson2/taste_class.png" width="800"> <figcaption>Figure 2.3. Taste classification engine </figcaption>
# </figure>
# 
# The resulting "rule" is a classification engine.
# 
# :::{note}
# I have implicitly assumed that there are quantitative metrics for softness and color (RGB+intensity perhaps), and we don't want to actually put one into our mouth.  An interesting twist on this example would be [Durian](https://en.wikipedia.org/wiki/Durian#/media/File:Durian.jpg), a most tasty fruit but in its harvested state most decidedly not soft nor of especially pleasing color.
# :::

# ## A Prediction Engine Example
# 
# Another simple example of machine learning (this time using numbers) is the mundane process of fitting a model to data; or in ML jargon the building of a prediction engine (the model equation) and subsequent application of the engine to new situations.
# 
# Consider a simple case where we have some observations like:
# 
# |predictor1|predictor2|response|
# |---:|---:|---:|
# | 0.0 | 1.0 | 0.0 |
# | 10.0 | 1.0 | 10.0 |
# | 20.0 | 1.0 | 20.0 |
# | 30.0 | 1.0 | 30.0 |
# | 40.0 | 1.0 | 40.0 |
# | 50.0 | 1.0 | 50.0 |
# | 60.0 | 1.0 | 60.0 |
# | 70.0 | 1.0 | 70.0 |
# | 80.0 | 1.0 | 80.0 |
# | 90.0 | 1.0 | 90.0 |
# | 100.0 | 1.0 | 100.0 |
# | 0.0 | 2.0 | 0.0 |
# | 10.0 | 2.0 | 5.0 |
# | 20.0 | 2.0 | 10.0 |
# | 30.0 | 2.0 | 15.0 |
# | 40.0 | 2.0 | 20.0 |
# | 50.0 | 2.0 | 25.0 |
# | 60.0 | 2.0 | 30.0 |
# | 70.0 | 2.0 | 35.0 |
# | 80.0 | 2.0 | 40.0 |
# | 90.0 | 2.0 | 45.0 |
# | 100.0 | 2.0 | 50.00 |
# | 0.0 | 6.0 | 0.0 |
# | 10.0 | 6.0 | 1.667 |
# | 20.0 | 6.0 | 3.333 |
# | 30.0 | 6.0 | 5.0 |
# | 40.0 | 6.0 | 6.667 |
# | 50.0 | 6.0 | 8.333 |
# | 60.0 | 6.0 | 10.0 |
# | 70.0 | 6.0 | 11.667 |
# | 80.0 | 6.0 | 13.333 |
# | 90.0 | 6.0 | 15.0 |
# | 100.0 | 6.0 | 16.667 |
# 
# And if we simply try plotting we don't learn much.

# In[1]:


input1=[10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
        10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
        10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,]
input2=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
        2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
        6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0]
output=[10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
        5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0, 
       1.6666833333333333, 3.3333666666666666, 5.00005, 6.666733333333333, 8.333416666666666, 10.0001, 11.666783333333331, 13.333466666666666, 15.00015, 16.666833333333333]

import matplotlib.pyplot as plt # the python plotting library
plottitle ='Simple Prediction Engine' 
mydata = plt.figure(figsize = (10,5)) # build a square drawing canvass from figure class
plt.plot(input1, output, c='red',linewidth=0,marker='o') 
plt.plot(input2, output, c='blue',linewidth=0,marker='o')
#plt.plot(time, accumulate, c='blue',drawstyle='steps') # step plot
plt.xlabel('Predictor')
plt.ylabel('Response')
plt.legend(['Predictor 1','Predictor 2'])
plt.title(plottitle)
plt.show()


# Lets postulate a prediction engine structure as
# 
# $$r=\beta_1 p_1^{\beta_2} \cdot \beta_3 p_2^{\beta_4}$$
# 
# and try to pick values that make the model explain the data - here strictly by plotting.
# 
# First our prediction engine 

# In[2]:


def response(beta1,beta2,beta3,beta4,predictor1,predictor2):
    response = (beta1*predictor1**beta2)*(beta3*predictor2**beta4)
    return(response)


# Now some way to guess model parameters (the betas) and plot our model response against the observed response.  Here we choose to plot the observed values versus model values, if we find a good model, the line should plot as equal value line (45 degree line)  

# In[3]:


beta1 = 1
beta2 = 1
beta3 = 1
beta4 = 1

howmany = len(output)
modeloutput = [0 for i in range(howmany)]

for i in range(howmany):
    modeloutput[i]=response(beta1,beta2,beta3,beta4,input1[i],input2[i])
    
# now the plot
plottitle ='Simple Prediction Engine' 
mydata = plt.figure(figsize = (10,5)) # build a square drawing canvass from figure class
plt.plot(modeloutput, output, c='red',linewidth=1,marker='o') 
#plt.plot(input2, output, c='blue',linewidth=0,marker='o')
#plt.plot(time, accumulate, c='blue',drawstyle='steps') # step plot
plt.xlabel('Model Value')
plt.ylabel('Observed Value')
#plt.legend(['Predictor 1','Predictor 2'])
plt.title(plottitle)
plt.show()


# Our first try is not too great.  Lets change $\beta_2$

# In[4]:


beta1 = 1
beta2 = 2
beta3 = 1
beta4 = 1

howmany = len(output)
modeloutput = [0 for i in range(howmany)]

for i in range(howmany):
    modeloutput[i]=response(beta1,beta2,beta3,beta4,input1[i],input2[i])
    
# now the plot
plottitle ='Simple Prediction Engine' 
mydata = plt.figure(figsize = (10,5)) # build a square drawing canvass from figure class
plt.plot(modeloutput, output, c='red',linewidth=1,marker='o') 
#plt.plot(input2, output, c='blue',linewidth=0,marker='o')
#plt.plot(time, accumulate, c='blue',drawstyle='steps') # step plot
plt.xlabel('Model Value')
plt.ylabel('Observed Value')
#plt.legend(['Predictor 1','Predictor 2'])
plt.title(plottitle)
plt.show()


# Not much help.  After enough trials we might stumble on:

# In[5]:


beta1 = 1
beta2 = 1
beta3 = 1
beta4 = -0.9

howmany = len(output)
modeloutput = [0 for i in range(howmany)]

for i in range(howmany):
    modeloutput[i]=response(beta1,beta2,beta3,beta4,input1[i],input2[i])
    
# now the plot
plottitle ='Simple Prediction Engine' 
mydata = plt.figure(figsize = (10,5)) # build a square drawing canvass from figure class
plt.plot(modeloutput, output, c='red',linewidth=1,marker='o') 
#plt.plot(input2, output, c='blue',linewidth=0,marker='o')
#plt.plot(time, accumulate, c='blue',drawstyle='steps') # step plot
plt.xlabel('Model Value')
plt.ylabel('Observed Value')
#plt.legend(['Predictor 1','Predictor 2'])
plt.title(plottitle)
plt.show()


# And that's not too bad.  What would help is some systematic way to automatically update the model parameters until we get a good enough prediction engine, then go out and use it.  To get to perfection (which we can in this example because I know the data source), if we set the first three parameters to 1 and the last to -1 we obtain:

# In[6]:


beta1 = 1
beta2 = 1
beta3 = 1
beta4 = -1

howmany = len(output)
modeloutput = [0 for i in range(howmany)]

for i in range(howmany):
    modeloutput[i]=response(beta1,beta2,beta3,beta4,input1[i],input2[i])
    
# now the plot
plottitle ='Simple Prediction Engine' 
mydata = plt.figure(figsize = (10,5)) # build a square drawing canvass from figure class
plt.plot(modeloutput, output, c='red',linewidth=1,marker='o') 
#plt.plot(input2, output, c='blue',linewidth=0,marker='o')
#plt.plot(time, accumulate, c='blue',drawstyle='steps') # step plot
plt.xlabel('Model Value')
plt.ylabel('Observed Value')
#plt.legend(['Predictor 1','Predictor 2'])
plt.title(plottitle)
plt.show()


# Now with our machine all learned up we can use it for other cases, for instance if the inputs are [121,2]

# In[7]:


print('predicted response is',response(beta1,beta2,beta3,beta4,121,2))


# ### Automating the Process
# 
# To complete this example, instead of us (humans) doing the trial and error, lets let the machine do the work.  For that we need a way to assess the 'quality' of our prediction model, a way to make new guesses, and a way to rank findings.  
# 
# A terribly inefficient way, but easy to script is a grid search.  We will take our 4 parameters and try combinations for values ranging between -1 and 1 and declare the best combination the model.  

# In[8]:


# identify, collect, load data

input1=[10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
        10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
        10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,]
input2=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
        2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
        6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0]
output=[10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
        5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0, 
       1.6666833333333333, 3.3333666666666666, 5.00005, 6.666733333333333, 8.333416666666666, 10.0001, 11.666783333333331, 13.333466666666666, 15.00015, 16.666833333333333]

# a prediction engine structure (notice some logic to handle zeros)
def response(beta1,beta2,beta3,beta4,predictor1,predictor2):
    if predictor1 == 0.0 and predictor2 ==0.0:
        response = (beta1*predictor1**abs(beta2))*(beta3*predictor2**abs(beta4))
    if predictor1 == 0.0 and predictor2 !=0.0:
        response = (beta1*predictor1**abs(beta2))*(beta3*predictor2**(beta4))
    if predictor1 != 0.0 and predictor2 ==0.0:
        response = (beta1*predictor1**(beta2))*(beta3*predictor2**abs(beta4))
    if predictor1 != 0.0 and predictor2 !=0.0:
        response = (beta1*predictor1**(beta2))*(beta3*predictor2**(beta4))
    else:
        response = 1e9
    return(response)
# a measure of model quality
def quality(observed_list,model_list): 
    if len(observed_list) != len(model_list):
        raise Exception("List lengths incompatable")
    sse = 0.0
    howmany=len(observed_list)
    for i in range(howmany):
        sse=sse + (observed_list[i]-model_list[i])**2
    return(sse)
# define search region 
index_list = [i/10 for i in range(-20,21,2)] # index list is -1.0,-0.9,-0.8,....,0.9,1.0

howmany = 0 # keep count of how many combinations
error   = 1e99          # a big value, we are trying to drive this to zero
xbest   = [-1,-1,-1,-1] # variables to store our best solution parameters
modeloutput = [0 for i in range(len(output))] # space to store model responses


# perform a search - here we use nested repetition
for i1 in index_list:
    for i2 in index_list:
        for i3 in index_list:
            for i4 in index_list:
                howmany=howmany+1 # keep count of how many times we learn
                beta1 = i1
                beta2 = i2
                beta3 = i3
                beta4 = i4

                for irow in range(len(output)): 
                    modeloutput[irow]=response(beta1,beta2,beta3,beta4,input1[irow],input2[irow])
                guess = quality(output,modeloutput) # current model quality
 #               print(guess)
                if guess <= error:
                    error = guess
                    xbest[0]= beta1
                    xbest[1]= beta2
                    xbest[2]= beta3
                    xbest[3]= beta4
print("Search Complete - Error Value ",round(error,8))
print("Combinations Examined : ",howmany)
print("Beta 1 ",xbest[0])
print("Beta 2 ",xbest[1])
print("Beta 3 ",xbest[2])
print("Beta 4 ",xbest[3])
for irow in range(len(output)): 
    modeloutput[irow]=response(xbest[0],xbest[1],xbest[2],xbest[3],input1[irow],input2[irow])

# now the plot
import matplotlib.pyplot as plt # the python plotting library
plottitle ='Simple Prediction Engine - Automated Learning \n'
plottitle = plottitle + 'Model Error =' + repr(round(error,8)) + '\n'
plottitle = plottitle + 'Beta =' + repr(xbest)
mydata = plt.figure(figsize = (10,5)) # build a square drawing canvass from figure class
plt.plot(modeloutput, output, c='red',linewidth=1,marker='o') 
#plt.plot(input2, output, c='blue',linewidth=0,marker='o')
#plt.plot(time, accumulate, c='blue',drawstyle='steps') # step plot
plt.xlabel('Model Value')
plt.ylabel('Observed Value')
#plt.legend(['Predictor 1','Predictor 2'])
plt.title(plottitle)
plt.show()


# ### Interpreting our results
# Suppose we are happy with the results, lets examine what the machine is telling us.
# 
# First if we examine the engine structure using the values fitted we have
# 
# - Our original structure before our engine got learned
# 
# $$r=\beta_1 p_1^{\beta_2} \cdot \beta_3 p_2^{\beta_4}$$
# 
# - After its all learned up!
# 
# $$r=1.0 p_1^{1.0} \cdot 1.0 p_2^{-1.0}$$
# 
# - Lets do a little algebra
# 
# $$r=\frac{p_1 }{p_2}$$
# 
# A little soul searching and we realize that $\beta_1 \cdot \beta_3$ is really a single parameter and not really two different ones.  If we knew that ahead of time, our seacrh region could be reduced.  At this point all we really wanted here is an example of ML to produce a prediction engine (exclusive of the symbolic representation).  We have done it the model is 
# 
# $$r=\frac{p_1 }{p_2}$$
# 
# If we were to use it for future response prediction, then a user interface is in order.  Something as simple as:

# In[9]:


newp1 = 119
newp2 = 2.1
newot = response(xbest[0],xbest[1],xbest[2],xbest[3],newp1,newp2)
print('predicted response to predictor1 =',newp1,'and predictor2 =',newp2,' is :',round(newot,3))


# ## heading title
# 
# Before we exit this section lets examine the last example a bit more.  We have a two-dimensional input structure, so we can conceivable plot the input space and the response into the 3-rd axis.  First just the observations.

# In[10]:


#!/usr/bin/python3

import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy.random import randn
from scipy import array, newaxis


input1=[10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
        10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
        10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,]
input2=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
        2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
        6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0]
output=[10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
        5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0, 
       1.6666833333333333, 3.3333666666666666, 5.00005, 6.666733333333333, 8.333416666666666, 10.0001, 11.666783333333331, 13.333466666666666, 15.00015, 16.666833333333333]

points = [[0,0,0] for i in range(len(output))]

for i in range(len(output)):
    points[i] = [input1[i],input2[i],output[i]]

DATA = numpy.array(points)

Xs = DATA[:,0]
Ys = DATA[:,1]
Zs = DATA[:,2]

#print(DATA)

# Creating figure
fig = plt.figure(figsize = (5, 5))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(Xs, Ys, Zs, color = "green")
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()


# Assuming the response is some smooth function of the input feature values, we could attempt a contour plot.
# ======
## plot:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show() # or:
# fig.savefig('3D.png')
# In[11]:


import pandas
my_xyz = pandas.DataFrame(DATA) # convert into a data frame
import numpy 
import matplotlib.pyplot
from scipy.interpolate import griddata
# extract lists from the dataframe
coord_x = my_xyz[0].values.tolist()
coord_y = my_xyz[1].values.tolist()
coord_z = my_xyz[2].values.tolist()
coord_xy = numpy.column_stack((coord_x, coord_y))
# Set plotting range in original data units
lon = numpy.linspace(min(coord_x), max(coord_x), 300)
lat = numpy.linspace(min(coord_y), max(coord_y), 300)
X, Y = numpy.meshgrid(lon, lat)
# Grid the data; use cubic spline interpolation (other choices are nearest and linear)
Z = griddata(numpy.array(coord_xy), numpy.array(coord_z), (X, Y), method='cubic', fill_value = 'nan')
# Build the map
matplotlib.pyplot.rcParams["figure.figsize"] = [12.00, 6.00]
matplotlib.pyplot.rcParams["figure.autolayout"] = True
fig, ax = matplotlib.pyplot.subplots()
levels=[5,10,15,20,25,30,35,40,45,50]
CS = ax.contour(X, Y, Z, levels=20, linewidths=3)
ax.plot([50,50],[0,7])
ax.plot([10,100],[5,5])
ax.plot(coord_x,coord_y,marker=".",linestyle="none",markersize=13)
ax.clabel(CS, inline=2, fontsize=12)
ax.set_title('Contour Plot from Gridded Data File')
ax.set_xlim([0,120])
ax.set_ylim([0,7])


# Conceivably, could use this as the predcition engine - for instance the plot suggests that at (50,5) the response should be near 5 (based on the plotted contours); notice (50,5) is NOT in the observation set - so this is truely a prediction.

# In[12]:


newp1 = 50
newp2 = 5
newot = response(xbest[0],xbest[1],xbest[2],xbest[3],newp1,newp2)
print('predicted response to predictor1 =',newp1,'and predictor2 =',newp2,' is :',round(newot,3))


# So whats going on with the prediction?  The contour plot has an implied data model (probably inverse distance - we would need to look at the internals of the gridding tool); our ML model has a different implied data model - we really don't know which one is better until we get a new observation, then in either case we would rerun the "training step" to update the model.
# 
# Now notice something our ML model can do that our contour plot does not do (easily) - that is to extrapolate beyond the input ranges smoothly.
# 
# For instance at (110,3) the contour plot shows us nothing at best the response will be in the 20s to 30s if we mentally extend the contour lines.  To the ML model its just another input/response combination:

# In[13]:


newp1 = 110
newp2 = 3
newot = response(xbest[0],xbest[1],xbest[2],xbest[3],newp1,newp2)
print('predicted response to predictor1 =',newp1,'and predictor2 =',newp2,' is :',round(newot,3))


# ## References
# 
# 
# 
# 

# In[ ]:





# In[ ]:




