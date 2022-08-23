#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Common Data Types
# 
# Two "kinds" of data that exist are: 
# 1. Quantitative 
# 2. Qualitative
# 
# ## Quantitative Data
# Quantitative data deals with numbers and refers to variables that can be mea-
# sured objectively.  The root of the word is "quantity" which is an expression of how much of something exists.
# 
# The load on a building, the width of a channel, the temperature of lake, and the bearing capacity of a soil are quantitative data. While the meaning of the quantitative data is clear and unambigious there is uncertainty associated with them from measurement errors, transcription errors, and related causes.
# 
# ## Qualitative Data
# Qualitative data refers descriptors that are subjective in nature. 
# The root of the work is "quality" which is an expression of how desirable is the object.
# 
# Descriptors such as *small* loads, *wide* channel, *sharp* turn and *extreme* drought
# are some examples of qualitative data. The subjectivity arises because what is a
# *small* load for a large structure can be a substantial load for a small structure.
# The meaning of the qualitative data largely depends upon the situation and its
# interpretation can vary widely from one person to the next. Qualitative variables
# sometimes can be made quantitative for analysis. For example, flood is a
# qualitative variable. However, we can define a threshold flowrate to determine
# the flooding and non-flooding states in a river section.
# 

# # Scale
# 
# Four common measurement scales of data are:
# 
# 1. nominal,
# 2. ordinal,
# 3. interval, and 
# 4. ratio scales.
# 
# ## Nominal Scale
# The nominal scale is the simplest of all data scales. 
# These data are qualitative and often used to label the data. 
# For example, a traffic survey could ask a respondent’s gender (i.e., whether they are ’Male’ or ’Female’). 
# The category ’Gender’ has two labels (Male and Female) which can be be used to filter data
# for further analysis.
# Nominal scale data have no order or hierarchy associated with them. 
# 
# 
# ## Ordinal Scale
# The ordinal scale involves arranging data in an order. This ordering can be used
# to rank or compare two or more values. Ordinal scale also refers to qualitative
# data. For example we could classify droughts as ’mild’, moderate’ and ’severe’.
# Clearly severe droughts have far greater impacts than moderate droughts which
# in turn have a greater impact than mild droughts.
# We can also assign numeric values (say, 1 for mild, 2 for moderate and 3 for se-
# vere) to rank drought events in a region. However, we cannot do any arithmetic
# on these data (mild + moderate 6 = severe). Similarly the difference between a
# mild and moderate drought is not the same as the difference between a moderate
# and severe drought.
# Ordinal scales are often used to assess relative perceptions, choices and feedback.
# 
# ## Interval Scale
# Interval scale data have an order and it is possible to calculate exact differences
# between the values. The interval scale is used with quantitative data. You can
# calculate summary measures (e.g., mean, variance) on interval scale data.
# The temperature measured in Celsius is a classic example of interval scale data.
# 20 oC is twice the value of 10 oC (i.e., the difference is 10 oC). Similarly, the
# difference between 90 oC and 80 oC is also 10 oC. This is because the values
# are being measured on the same interval.
# However, 20 oC is not twice as hot as 10 oC. This is evident when these
# temperatures are converted to the Fahrenheit scale (10 oC = 50 oF and 20 oC
# = 68 oF ). The reason for this is that there is no absolute zero in either Celsius
# of Fahrenheit scales. Zero just another measurement.
# 
# ## Ratio Scale
# Variables measured on the ratio scale have an order, there are measured on
# fixed intervals (differences between two values can be calculated) and have an
# absolute zero. In other words, Ratio Scale has all the properties of an interval
# scale and also has an absolute zero. Ratio scale is used for quantitative data.
# Consider daily rainfall amounts. There is an absolute zero which corresponds to
# no rainfall. Other measurable amounts of rainfall are measured with respect to
# this absolute zero. 20’ of rainfall is more than 10’ of rainfall. Also, 20’ of rainfall
# is twice the amount of 10’ accumulation. Their ratio is equal to 2. In a similar
# vein, 3’ of rainfall is twice the amount of 6’ accumulation (again the ratio is
# 2). As the name suggests, variables measured on ratio scale can be divided to
# obtain ratios or the degree of proportionality between two observations.
# 3
# 

# # Continuous and Discrete Data
# Quantitative data can be further split into continuous and discrete data.
# Discrete data are integers and refer to variables that measure whole or indivisible entities. 
# The number of vehicles passing an intersection over a year is discrete data as this is a whole number (integer).
# 
# Continuous data on the other hand are not restricted to integer values but also can take on fractional or decimal values. For example, The temperature of water in a lake can be 67.3$^o$F .
# Sometimes a variable that is continuous can be discretely measured or made
# discrete. For example, if we measure (or round off) the temperature of a lake
# to the nearest degree then we would have temperature measurements that are
# only integers. While continuous variables can be treated as discrete due to
# measurement limitations or data approximations, it is important to remember
# that intrinsically they can take continuous values and the discretization was
# simply a matter of convenience or a measurement limitation.
# 
# Summary measures of discrete variables (e.g., mean, standard deviation) can
# sometime have decimal values. For example, if 3 cars pass an intersection in
# the first hour and 4 cars in the second then the average number of cars is 3.5.
# In such instances, the value must be rounded appropriately (say to 4 if we are
# using it to estimate design loads on the pavement) as the decimal number is not
# physically realistic.
# 
# ## Continuous on a real line
# Some variables in civil engineering are continuous on a real line (axis). For
# example, soil temperature measured in Celsius or Fahrenheit can assume both
# positive and negative values in cold regions.
# Similarly, the groundwater table measured with respect to mean sea level (MSL)
# can be either positive (i.e., below MSL) or negative (above MSL). A zero value
# simply suggests that the water table coincides with MSL which is just as a valid
# state as water table being above or below MSL.
# For variables that are continuous on a real line (axis), zero values have no special
# meaning and represent yet another value that the variable can take. In other
# words, they are often measured on an interval scale.
# 
# ## Positive Continuous Variables
# As discussed above, the magnitude of a force is always positive. As this magni-
# tude can take decimal values it is a positive continuous variable. Many variables
# of interest to civil engineers are positive continuous. Flowrates, pollutant con-
# centrations, the magnitude of shear stresses are all positive continuous.
# Some variables that are continuous on a real line can also be made positive con-
# tinuous by shifting the datum. For example, there are no negative temperatures
# on Kelvin scale. Temperatures measured using Celsius scale can be made all
# positive by converting them into Kelvin scale. In a similar vein, the groundwa-
# ter table measurements at a well can be made all positive by measuring values
# from the top of the well casing (assuming positive downwards) instead of the
# mean sea level.
# 
# # Mixture Datasets
# The positive continuous scale starts at zero and in theory extends all the way to
# infinity. For some positive continuous variables zero is yet another value if they
# are measured on the interval scale. On the other hand if they are measured on
# a ratio scale the value of zero is absolute and can assume special importance.
# For example, zero rainfall implies no rainfall. Therefore, a time-series data of
# daily rainfall in a year can have several values of zero (no rainfall) and some days
# with positive (non-zero rainfall). Such a dataset can be viewed as a mixture
# - At a high level the data is discrete where each day belongs to either (rain-
# fall=Yes) or (rainfall=No) state. Non-zero, positive continuous values exist for
# (Rainfall=Yes) state (see Figure 1).

# In[ ]:




