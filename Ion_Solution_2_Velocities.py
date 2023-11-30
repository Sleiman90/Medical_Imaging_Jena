#!/usr/bin/env python
# coding: utf-8

# We start by defining the necessary imports:

# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# nicer default plot settings
plt.rcParams.update({'font.size': 20, 
                     'lines.linewidth': 2, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'axes.grid': True, 
                     'figure.figsize': (12,7)})


# Calculating the electron velocities for the given voltage range can be done by using the solutions from **(b)** and **(c)**

# In[9]:


U_a = np.linspace(0,1000,1000)
v_clas = np.sqrt(2 * U_a / 511) # classical non-relativistic solution
v_rel = np.sqrt(1 - (U_a/511 + 1) ** (-2)) # relativistic solution


# Finally, plotting the electron velocities shows that when not taking relativistic effects into account the electron speed would quickly surpass the speed of light. This is physically not possible! When using the proper relativistic solution that takes the electrons rest mass into account the calculated electron velocity will rather approach the speed of light and never surpass it.

# In[10]:


plt.plot(U_a, v_clas, label = 'classical')
plt.plot(U_a, v_rel, label = 'relativistic')

# nicer plot
plt.hlines(1, U_a[0], U_a[-1], linestyles = 'dashed')
plt.text(U_a[-1], 1, 'speed of light', ha='right', va='bottom')
plt.legend()
plt.xlabel('Acceleration voltage [kV]')
plt.ylabel('Electron velocity / c')
plt.xlim(U_a[0], U_a[-1])
plt.ylim(0, 2)
plt.grid(True, which='minor', linestyle='--', alpha=0.4)

