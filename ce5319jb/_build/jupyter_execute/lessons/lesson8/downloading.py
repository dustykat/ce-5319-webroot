#!/usr/bin/env python
# coding: utf-8

# # Downloading Remote Data
# 
# For machine learning to be useful, it needs data, and large quantities (expressed as records in a database - millions of records; expressed as bytes thousands of terabytes).  Manual collection of such data for data science is impractical, so we want to automate where practical.
# 
# There are a lot of data science blog posts with databases you can download usually as a single compressed file (.zip, .tar, ....); however these examples hide the actual workflow of a data science project.  More likely you will have to visit (automated) multiple websites, identify a collection of useful files then get them to your machine for processing.  In this section some examples are presented mostly as an archive of potentially useful techniques.
# 
# :::{warning}
# Good luck if these work for you as is, you will have to tinker until you can automate the process reliably. Be careful with destination directories that you don't clobber your own file system when doing recursive traverse of a remote website.  
# :::
# 

# ## Tools for use
# 
# A few useful modules are `requests` and `wget`; there are a few others `curl` comes to mind.  In Jupyter Notebooks downloading single files is relatively straightforward; downloading multiple files is tricky, but the most realistic representation of actual workflow.  
# 
# For multiple files I resort to operating system calls using various Magic Functions as are illustrated.  The examples herein are assuming a Linux (Debian-based) operating system, so Windoze users will have to find equivalents (or use the WSL feature in Windoze)

# 

# ## Example 1 Evaporation Trend Examination
# 
# Keep in mind the example herein is about data acquisition.
# 
# ### Background
# Global warming is a currently popular and hotly (pun intended) debated issue. 
# The usual evidence is temperature data presented as a time series with various temporal correlations to industrial activity and so forth. The increase in the global temperature is not disputed - what it means for society and how to respond is widely disputed.
# 
# One possible consequence of warming, regardless of the cause is an expectation that
# <strong>evaportation rates would increase</strong> and temperate regions would experience more
# drought and famine, and firm water yields would drop. 
# 
# However in a paper by Peterson and others (1995) the authors concluded from analysis of pan evaporation data in various parts of the world, that there has been a <strong>downward trend in evaporation</strong> at a significance level of 99%. 
# Pan evaporation is driven as much by direct solar radiation (sun shining on water) as by surrounding air temperature.
# 
# Global dimming is defined as the decrease in the amounts of solar radiation reaching the surface of the Earth. The by-product of fossil fuels is tiny particles or pollutants which absorb solar energy and reflect back sunlight into space. This phenomenon was first recognized in the year 1950. Scientists believe that since 1950, the sun’s energy reaching Earth has dropped by 9% in Antarctica, 10% in the USA, 16% in parts of Europe and 30% in Russia – putting the overall average drop to be at an enormous 22%. This causes a high risk to our environment.
# 
# Aerosols have been found to be the major cause of global dimming. The burning of fossil fuels by industry and internal combustion engines emits by-products such as sulfur dioxide, soot, and ash. These together form particulate pollution—primarily called aerosols. Aerosols act as a precursor to global dimming in the following two ways:
# 
# These particle matters enter the atmosphere and directly absorb solar energy and reflect radiation back into space before it reaches the planet’s surface.
# Water droplets containing these air-borne particles form polluted clouds. These polluted clouds have a heavier and larger number of droplets. These changed properties of the cloud – such clouds are called ‘brown clouds’ – makes them more reflective.
# Vapors emitted from the planes flying high in the sky called contrails are another cause of heat reflection and related global dimming.
# 
# Both global dimming and global warming have been happening all over the world and together they have caused severe changes in the rainfall patterns. It is also believed that it was global dimming behind the 1984 Saharan drought that killed millions of people in sub-Saharan Africa. Scientists believe that despite the cooling effect created by global dimming, the earth’s temperature has increased by more than 1 deg. in the last century.
# 
# ### References
# 
# [Peterson, T.C., Golubev, V.S. and Groisman, P. Ya. 1995. Evaporation
# losing its strength. Nature 377: 687-688.](http://54.243.252.9/ce-5319-webroot/ce5319jb/lessons/lesson8/Peterson-Nature1995-377.pdf)
# 
# https://www.conserve-energy-future.com/causes-and-effects-of-global-dimming.php
# 
# ## Example Data Science Problem 
# In Texas, evaporation rates (reported as inches per month) are available from the Texas Water Development Board.
# https://waterdatafortexas.org/lake-evaporation-rainfall
# The map below shows the quadrants (grid cells) for which data are tabulated.
# 
# ![figure1](EvapMap.png)
# 
# Cell '911' is located between Corpus Christi and Houston in the Coastal Plains of Texas.  A copy of the dataset downloaded from the Texas Water Development Board is located at http://www.rtfmps.com/share_files/all_quads_gross_evaporation.csv
# 
# Using naive data science anlayze the data for Cell '911' and decide if the conclusions by Peterson and others (1995) are supported by this data.
# 

# ### Exploratory Analysis
# To analyze these data a first step is to obtain the data.  The knowlwdge that the data are arranged in a file with a ``.csv`` extension is a clue how to proceede.  We will need a module to interface with the remote server, in this example lets use ``requests`` , which allows us to make GET and POST requests using the HTTP/HTTPS protocols to interact with web servers. So first we load the module

# In[1]:


import requests # Module to process http/https requests


# Now we will generate a ``GET`` request to the remote http server.  I chose to do so using a variable to store the remote URL so I can reuse code in future projects.  The ``GET`` request (an http/https method) is generated with the requests method ``get`` and assigned to an object named ``rget`` -- the name is arbitrary.  Next we extract the file from the ``rget`` object and write it to a local file with the name of the remote file - esentially automating the download process. Then we import the ``pandas`` module.

# In[2]:


remote_url="http://54.243.252.9/ce-5319-webroot/ce5319jb/lessons/lesson8/all_quads_gross_evaporation.csv"  # set the url
response = requests.get(remote_url, allow_redirects=True)  # get the remote resource, follow imbedded links
open('all_quads_gross_evaporation.csv','wb').write(response.content) # extract from the remote the contents, assign to a local file same name


# In[3]:


import pandas as pd # Module to process dataframes
import matplotlib.pyplot 


# In[4]:


ewhat easier than using primatives, and gives graphing tools)


# Now we can read the file contents and check its structure, before proceeding.

# In[6]:


#evapdf = pd.read_csv("all_quads_gross_evaporation.csv",parse_dates=["YYYY-MM"]) # Read the file as a .CSV assign to a dataframe evapdf
evapdf = pd.read_csv("all_quads_gross_evaporation.csv") # Read the file as a .CSV assign to a dataframe evapdf

evapdf.head() # check structure


# Structure looks like a spreadsheet as expected; lets plot the time series for cell '911'

# In[7]:


evapdf.plot.line(x='YYYY-MM',y='911') # Plot quadrant 911 evaporation time series 


# Now we can see that the signal indeed looks like it is going up at its mean value then back down. Lets try a moving average over 12-month windows. The syntax is a bit weird, but it should dampen the high frequency (monthly) part of the signal.  Sure enough there is a downaward trend at about month 375, which we recover the date using the index -- in this case around 1985.
# 

# In[8]:


movingAvg=evapdf['911'].rolling(12, win_type ='boxcar').mean()
movingAvg
movingAvg.plot.line(x='YYYY-MM',y='911')
evapdf['YYYY-MM'][375]


# So now lets split the dataframe at April 1985.  Here we will build two objects and can compare them.  Notice how we have split into two entire dataframes.

# In[9]:


evB485loc = evapdf['YYYY-MM']<'1985-04'  # filter before 1985
evB485 = evapdf[evB485loc]
ev85uploc = evapdf['YYYY-MM']>='1985-04' # filter after 1985
ev85up= evapdf[ev85uploc]
print(evB485.head())
print(ev85up.head())


# Now lets get some simple descriptions of the two objects, and we will ignore thay they are time series.

# In[10]:


evB485['911'].describe()


# In[11]:


ev85up['911'].describe()


# If we look at the means, the after 1985 is lower, and the SD about the same, so there is maybe support of the paper claims, but the median has increased while the IQR is practically unchanged.  We can produce boxplots from the two objects and see they are different, but not by much.  So the conclusion of the paper has support but its pretty weak and hardly statisticlly significant. 

# In[20]:


evB485['911'].plot.box()


# In[21]:


ev85up['911'].plot.box()


# At this point, we would appeal to some kind of hypothesis testing or some other serious statistical analysis tools.  For example a non-paramatric test called the ``mannwhitneyu`` test is pretty quick to implement 
# 
# ### Background
# In statistics, the Mann–Whitney U test (also called the Mann–Whitney–Wilcoxon (MWW), Wilcoxon rank-sum test, or Wilcoxon–Mann–Whitney test) is a nonparametric test of the null hypothesis that it is equally likely that a randomly selected value from one population will be less than or greater than a randomly selected value from a second population.
# 
# This test can be used to investigate whether two independent samples were selected from populations having the same distribution.
# 
# ## Application
# As usual we need to import necessary tools, in this case scipy.stats.  Based on the module name, it looks like a collection of methods (the dot ``.`` is the giveaway).  The test itself is applied to the two objects, if there is a statistical change in behavior we expect the two collections of records to be different.

# In[26]:


from scipy.stats import mannwhitneyu # import a useful non-parametric test
stat, p = mannwhitneyu(evB485['911'],ev85up['911'])

print('statistic=%.3f, p-value at rejection =%.3f' % (stat, p))
if p > 0.05:
	print('Difference in Mean Values',round(evB485['911'].mean()-ev85up['911'].mean(),3),'is not significant')
else:
	print('Difference in Mean Values',round(evB485['911'].mean()-ev85up['911'].mean(),3),'is SIGNIFICANT')


# If there were indeed a 99% significance level, the p-value should have been smaller than 0.05 (two-tailed) and the p-value was quite high.  I usually check that I wrote the script by testing he same distribution against itself, I should get a p-value of 0.5.  Indeed that's the case.  

# In[29]:


stat, p = mannwhitneyu(evB485['911'],evB485['911'])
print('statistic=%.3f, p-value at rejection =%.3f' % (stat, p))
if p > 0.05:
	print('Difference in Mean Values',round(evB485['911'].mean()-evB485['911'].mean(),3),'is not significant')
else:
	print('Difference in Mean Values',round(evB485['911'].mean()-evB485['911'].mean(),3),'is SIGNIFICANT')


# Now lets repeat the analysis but break in 1992 when Clean Air Act rules were slightly relaxed:

# In[31]:


evB492loc = evapdf['YYYY-MM']<'1992'  # filter before 1992
evB492 = evapdf[evB492loc]
ev92uploc = evapdf['YYYY-MM']>='1992' # filter after 1992
ev92up= evapdf[ev92uploc]
#print(evB492.head())
#print(ev92up.head())


# In[32]:


stat, p = mannwhitneyu(evB492['911'],ev92up['911'])
print('statistic=%.3f, p-value at rejection =%.3f' % (stat, p))
if p > 0.05:
	print('Difference in Mean Values',round(evB492['911'].mean()-ev92up['911'].mean(),3),'is not significant')
else:
	print('Difference in Mean Values',round(evB492['911'].mean()-ev92up['911'].mean(),3),'is SIGNIFICANT')


# So even considering the key date of 1992, there is insufficient evidence for the claims (for a single spot in Texas), and one could argue that the claims are confounding -- as an FYI this eventually was a controversial paper because other researchers obtained similar results using subsets (by location) of the evaporation data.
# 

# In[ ]:





# In[ ]:




