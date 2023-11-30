#!/usr/bin/env python
# coding: utf-8

# As python3 libraries we only need numpy and matplotlib

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# nicer default plot settings
plt.rcParams.update({'font.size': 20, 'lines.linewidth': 2})


# We start with the definition of the x-values, which are used for the calculation of the $sin(x)$ and $cos(x)$ functions. For this purpose, we use the ```linspace()``` function, which allows us to specify a start value (first parameter) and an end value (second parameter) and will then distribute the given number of elements (third parameter) equally between them:

# In[3]:


x = np.linspace(0, 4 * np.pi, 1000)
print(f'Size/shape of x = {x.shape}')


# Now we can use the x-array to calculate $sin(x)$ and $cos(x)$ for all elements:

# In[4]:


y1 = np.sin(x)
y2 = np.cos(x)


# Finally, we plot the result

# In[5]:


fig, axes = plt.subplots(ncols=1, figsize=(10,8)) # create figure
axes.plot(x, y1) # plot y1 (sin) over x
axes.plot(x, y2) # plot y2 (cos) over x
axes.set_xlabel('x')
axes.set_ylabel('Function values')
axes.set_xlim(x[0],x[-1]) # set limits to the first and last element in x
axes.legend(('sin(x)','cos(x)'), loc='lower right')

# nicer plot format
axes.minorticks_on()
axes.grid(True, which='major', linestyle='-')
axes.grid(True, which='minor', linestyle='--', alpha=0.4)
axes.tick_params(labelsize=20)


# Because the creation of nice plots can be quite cumbersome, let's go through the commands for plotting again, this time step by step:
# 
# 1) First we create a new figure and within this figure a single axis for plotting (```ncols=1```). The figure will be empty without any content:

# In[6]:


fig, axes = plt.subplots(ncols=1, figsize=(10,8)) # create figure


# 2) To fill the figure, we add the actual plots to the previously obtained axes. Note that in matplotlib multiple ```plot()``` calls always draw into the same figure overlaying all plots: 

# In[7]:


axes.plot(x, y1) # plot y1 (sin) over x
axes.plot(x, y2) # plot y2 (cos) over x
fig # used for demonstration to show the updated figure


# 3) Let us add labels and a legend:

# In[8]:


axes.set_xlabel('x')
axes.set_ylabel('Function values')
axes.legend(('sin(x)','cos(x)'), loc='lower right')
fig


# 4) You may have noticed that the range of the x-axis extends beyond the data range in our ```x``` variable. We can use the ```set_xlim()``` method to set matching limits for the x-axis. Note that the syntax ```x[-1]``` selects the last element in an array.

# In[9]:


axes.set_xlim(x[0],x[-1]) # set limits to the first and last element in x
fig


# 5) Often a major and minor grid in the background can improve the readability of a plot. The following code snippet first enables minor ticks on the x-axis, then adds a major and a minor grid with different line styles as visibility (alpha) and finally sets a custom font-size:

# In[9]:


axes.minorticks_on()
axes.grid(True, which='major', linestyle='-')
axes.grid(True, which='minor', linestyle='--', alpha=0.4)
axes.tick_params(labelsize=20)
fig

