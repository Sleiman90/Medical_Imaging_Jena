#!/usr/bin/env python
# coding: utf-8

# As Python3 libraries we only need NumPy and matplotlib

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# nicer default plot settings
plt.rcParams.update({'font.size': 20, 
                     'lines.linewidth': 2, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'axes.grid': True, 
                     'figure.figsize': (12,7)})


# We start with the definition of variables and quantities to be used for the calculation

# In[3]:


k = 1.381E-23 # Boltzmann constant [J · K^-2]
eV = 1.602E-19 # Electron volt [J · eV^-1]

# Richardson's constant for [tungsten, thoriated tungsten and cesium on tungsten] 
# in [A · cm^-2 · K^-2]
Am = np.array([60, 3.0, 3.2])

# Work function for [tungsten, thoriated tungsten and cesium on tungsten] in [eV]
WA_eV = np.array([4.5, 2.8, 1.4])
WA_J = WA_eV * eV # Work function in [J]


# Next we define the temperature ranges in a vector by using the linspace() function

# In[4]:


# Create linearily spaced vector for the temperatures
TVec = np.linspace(1000,3000,1000)
TVec = TVec.reshape(TVec.size, 1) # give it a singleton dimension


# The tasks can be solved simultaneously by ensuring that all vectors containing constants (Am, WA_J) have the same orientation as a row vector, while the temperature vector TVec is a column vector. To achieve this, the orientation of Am and WA_J is transposed:

# In[5]:


WA_J_trans = WA_J.reshape((1, WA_J.size))
Am_trans = Am.reshape((1, Am.size))
print(f'WA_J_trans.shape = {WA_J_trans.shape}')
print(f'Am_trans.shape = {Am_trans.shape}')
print(f'TVec.shape = {TVec.shape}')


# Now, since the temperature vector ```TVec``` and the transposed vectors for the work function ```WA_J_trans``` and the Richardson's constant ```Am_trans``` have different orientations, using [Numpy Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) we can calculate equation (1)

# In[6]:


J = Am_trans * TVec ** 2 * np.exp(-WA_J_trans / (k * TVec))


# Finally, we plot the result

# In[7]:


fig, axes = plt.subplots(ncols=1, figsize=(10,8))
axes.plot(TVec, J)
axes.set_xlabel('Temperature [K]')
axes.set_ylabel('Current density [$A/cm^2$]')
axes.set_yscale('log')
axes.set_xlim(TVec[0, 0], TVec[-1, 0])
axes.legend(('W','W-Th','W-Cs'))
axes.grid(True, which='minor', linestyle='--', alpha=0.4)


# To get a sense of the magnitude of the current densities emitted by different materials, we compare the current densities of tungsten and caesium on tungsten for a a cathode temperature of 2500 K. Since we are only only discrete values for the temperatures, our ```T_Vec``` array may not include a value at exactly 2500 K. For simplicity, we can select the closest temperature to 2500 K by calculating the absolute difference between ```T_Vec``` and our reference temperature of 2500 K and applying [argmin()](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html)

# In[9]:


T_comp = 2500
T_ind = np.abs(TVec - T_comp).argmin()
print(f"For a cathode temperature of {T_comp} K, the current",
      f"density emitted from a cesium on tungsten cathode",
      f"is {J[T_ind,2] / J[T_ind,0]:.0f} times higher than",
      f"that emitted from a pure tungsten cathode")

