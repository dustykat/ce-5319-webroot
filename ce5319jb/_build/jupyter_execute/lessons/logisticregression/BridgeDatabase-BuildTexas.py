#!/usr/bin/env python
# coding: utf-8

# # Extract from NBI Database a specific state record

# In[1]:


get_ipython().system(' pwd')


# Actual code below; all intentionally suppressed for JupyterBook build to prevent attempted script run on missing files.
# 
# ```
# # Read bridge database line-by-line, extract state code 48, write result to a subset
# local_file_name='2021AllRecordsDelimitedAllStates.txt'
# # Connect to the file
# externalfile = open(local_file_name,'r') # create connection to file, set to read (r), file must exist
# ```

# ```
# records = [] # empty list to store the lines of the file
# linesread = 0 # counter for lines read
# howmanytoread = 800_000 # max number to read
# 
# for i in range(howmanytoread):
#     linenow = externalfile.readline()# read a line 
# # test for EOF
#     if not linenow: # if line is empty
#         print("End Of File detected")
#         break # end of file is reached, break out of the loop
#     records.append(linenow.rstrip().split(",")) # parse the line and insert into records
#     linesread = linesread + 1
# print(linesread)
# externalfile.close()
# # records contains all the lines
# ```

# ```
# #render as dataframe
# import pandas as pd
# alldata = pd.DataFrame(records)
# alldata.rename(columns=alldata.iloc[0], inplace = True)
# ```

# ```
# # verify structure
# alldata.tail()
# ```

# ```
# # select state code; Texas is 48
# texasdata = alldata.loc[alldata['STATE_CODE_001']=='48']
# ```

# ```
# # verify selection
# texasdata.head()
# ```

# ```
# # now write the reduced set to a file 
# texasdata.to_csv("2021TexasNBIData.csv", index = False)
# ```

# In[ ]:




