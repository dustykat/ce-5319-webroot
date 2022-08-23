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

# In[ ]:




