#!/usr/bin/env python
# coding: utf-8

# As python3 libraries we use numpy and matplotlib for numerical calculations and plotting as well as the constants package from scipy. We also set some nicer plot settings and set python to ignore warnings on division by zero

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

# nicer default plot settings
plt.rcParams.update({'font.size': 20, 
                     'lines.linewidth': 2, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'axes.grid': True, 
                     'figure.figsize': (12,7)})

# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore');


# ### Calculation of the intensity distribution

# The continuous spectral intensity distribution $I(\lambda)$ of Bremsstrahlung radiation produced by an electron hitting a solid target in an X-ray tube can be calculated using the following equation
# $$
#     I(\lambda) = K \big(\frac{\lambda}{\lambda_\mathrm{min}} - 1\big) \frac{1}{\lambda^3},
# $$
# 
# Since we want to calculate $I(\lambda)$ multiple times for different voltages, we will wrap the calculation in a function that takes a wavelength vector and a voltage as input and returns the intensity distribution $I(\lambda)$. In this function we also calculate the required $\lambda_{min}$ for the corresponding voltage from
# 
# $$
#     \lambda_\mathrm{min} = \frac{h \cdot c}{e \cdot U} = \frac{1.24\cdot10^{-6}\text{Vm}}{U\,[\text{V}]}.
# $$
# 
# To enter the values for the Planck constant $h$, the speed of light $c$, and electron charge $e$, we use of the submodule [constants](https://docs.scipy.org/doc/scipy/reference/constants.html) from the ``scipy`` package. For example, to access $e$, we can simply write:

# In[2]:


print(constants.e)


# Finally, let's put together the function for calculating the Bremsstrahlung intensity distributions:

# In[3]:


def bremsstrahlung(wavelength, voltage):
    k = 1 # set the constant k to one
    
    # calculate lambda_min
    hce = constants.h * constants.c / constants.e # factor in lambda_min [Vm]
    lambda_min_m = hce / voltage # the result in meters
        
    # calculate I(lambda)
    I = k * (wavelength / lambda_min_m - 1) / wavelength**3
        
    # allow only for positive values in I 
    # by applying the elementwise maximum between I and 0
    I = np.maximum(I, 0)
        
    return I


# Please note that before returning the calculated $I(\lambda)$, we apply the ``np.maximum(I, 0)`` function to all elements to allow only positive values for I. This is because the equation for $\lambda < \lambda_{min}$ returns negative spectral intensities, which is physically impossible.

# ### Bremsstrahlung for multiple voltages

# Since we want to calculate $I(\lambda)$ for a continuous distribution of wavelengths, we first create a vector ranging from 0 to 5 Angstroms. We then covert the units of the wavelengths to meters as this is the unit expected in our ```bremsstrahlung``` function. Finally, we also create a vector containing the 4 voltages that will be used for plotting:

# In[4]:


lambda_angstrom = np.linspace(0, 5, 1000) # in Angstrom
lambda_m = lambda_angstrom * 1e-10 # convert to SI
u_v = np.array([20, 30, 40, 50]) * 1e3


# > You might be tempted to create a variable called ``lambda`` for the wavelength, but note that in python this is a keyword reserved to define an anonymous function, so you'll get an error, if you try to assign anything to it.
# 
# Finally, we calculate the spectral intensity distrubtions of the bremsstrahlung for all voltages and plot the result:

# In[5]:


# loop over all voltages
for voltage in u_v:
    spectrum = bremsstrahlung(lambda_m, voltage)        
    
    # plot over lambda in angstrom,
    # also add a label with the voltages in keV and
    # format without decimal signs (.0f)) 
    plt.plot(lambda_angstrom, spectrum, label=f"{voltage*1e-3:.0f}")
    
# format the plot
plt.xlabel(r"$\lambda$ [Å]")
plt.ylabel(r"$I(\lambda)$ [a.u.]")
plt.legend(title="$U_A$ [kV]")
plt.xlim(lambda_angstrom[0], lambda_angstrom[-1])
plt.grid(True, which='minor', linestyle='--', alpha=0.4)

