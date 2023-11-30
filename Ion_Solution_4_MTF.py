#!/usr/bin/env python
# coding: utf-8

# Again, we start by importing numpy and matplotlib and by setting additional default plotting options. Additionally, we import the [loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) function and the [signal processing](https://docs.scipy.org/doc/scipy/reference/signal.html) toolbox from the SciPy package to load the input image. 

# In[2]:


import numpy as np
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt

# nicer default plot settings
plt.rcParams.update({'font.size': 20, 
                     'lines.linewidth': 2, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'axes.grid': True, 
                     'figure.figsize': (12,7)})


# First, we use [loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) to load the provided data (the data is provided as a .mat MATLAB file so that both students working with MATLAB and Python can easily load the data). 

# In[3]:


# load data
file_data = loadmat('../assignments/Ion_Assignment_4_MTFData_Sin.mat')
print(file_data.keys())


# The loaded MATLAB file contains a dictionary with ```'key' = value``` pairs. By printing the keys, we can see that besides some generic parameters such as ```['__header__', '__version__', '__globals__']``` it contains two variables ```['data', 'x']```, which is the data we want to get. Next, we extract these two variables by their key name, convert them to a NumPy array and also apply the [squeeze](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html) function to get rid of singleton dimensions:

# In[4]:


# extract the sinusoidal functions and the coordinates images
data = np.squeeze(np.array(file_data['data']))
x = np.squeeze(np.array(file_data['x']))


# We simply estimate the cycles per millimeter visually from the graph and add them as text labels. One cycle is defined as overlayed in red, which consequently results in 10 cy/mm, 5 cy/mm, 2.5 cy/mm and 1 cy/mm for the given sinusoidal functions. Adding a grid and especially a minor grid is very helpful here to estimate the cycles count per millimeter:

# In[5]:


plt.plot(x, data)
plt.plot(x[7500:8500], data[7500:8500], color='red')
plt.title('Input Sinusoidals')
plt.xlabel('x [mm]')
plt.ylabel('Intensity [a.u.]')
plt.xlim(x[0], x[-1])
plt.ylim(-1.05, 1.35)
plt.grid(True, which='major', linestyle='-')
plt.grid(True, which='minor', linestyle='--', alpha=0.4)
plt.text(1.25, 1.1, '10 cy/mm', ha='center')
plt.text(3.5, 1.1, '5 cy/mm', ha='center')
plt.text(5.75, 1.1, '2.5 cy/mm', ha='center')
plt.text(8.5, 1.1, '1 cy/mm', ha='center')


# Next, we define a Gaussian PSF and, for illustration, also calculate the MTF from it by using a Fourier transform:

# In[6]:


psf = signal.windows.gaussian(len(data), std=50)
mtf = np.fft.fftshift(np.abs(np.fft.fft(psf)))
mtf = mtf / np.sum(psf) # normalize

fig, ax = plt.subplots(ncols = 2, nrows = 1)
ax[0].plot(x, psf)
ax[0].set_title('PSF')
ax[0].set_xlabel('Position [mm]')
ax[0].set_xlim(x[0], x[-1])

freq = np.fft.fftshift(np.fft.fftfreq(len(x),x[1] - x[0]))
ax[1].plot(freq[int(len(x)/2):], mtf[int(len(x)/2):])
for cy in np.array([10, 5, 2.5, 1]):
    ind = np.argmin(np.abs(freq - cy))
    ax[1].plot(freq[ind], mtf[ind], markersize = 10, marker='o', label=f'{cy} cy/mm')
ax[1].set_xlim(0, 10)
ax[1].set_title('MTF')
ax[1].set_xlabel('Spatial Frequency [cy/mm]')
ax[1].legend()

for a in ax:
    a.grid(True, which='major', linestyle='-')
    a.grid(True, which='minor', linestyle='--', alpha=0.4)


# The point spread function (PSF) of an imaging system is the image of a point source at its input, where the dimensions of the point source are much smaller than the spatial resolution capability of the imaging system. Under these conditions, the source is essentially a mathematical point. If the imaging system is linear and shift-invariant, convolution of an arbitrary input function with the system's PSF gives the final image at the output of the imaging system. The modulation transfer function (MTF) of an imaging system is the equivalent description in the spatial frequency domain. It describes how much contrast (or modulation) in the original object is maintained by the imaging system. In other words, it characterizes how faithfully the spatial frequency content of the object gets transferred to the image. Mathematically, the MTF can be obtained from the absolute value of the normalized Fourier transform of the PSF.   
# 
# To apply the PSF to the entire input data, we need to perform a convolution between both arrays: 

# In[7]:


data_sampled = signal.convolve(psf, data, mode='same')


# Please note that we have to use ```mode='same'``` as optional parameter in order to only take the central part of the convolution result which has the same shape as the input ```data``` data. 
# 
# Finally, we plot the result and overlay it with the input. The result shows that the fine sinusoidal of 10 cy/mm can no longer be resolved at all, in the MTF shown above it is below the 10% threshold as described in the lecture. Although the sinusoidals of 2.5 cy/mm and 5 cy/mm can still be resolved, they suffer from an increasing loss in modulation and thus worsening contrast transfer by the imaging system.

# In[8]:


plt.subplots(ncols = 1, nrows = 1, figsize=(10,6))
plt.plot(x, data, label = 'Input', linewidth=1)
plt.plot(x[200:2350], data_sampled[200:2350] / np.sum(psf), label = 'Output: 10 cy/mm')
plt.plot(x[2350:4700], data_sampled[2350:4700] / np.sum(psf), label = 'Output: 5 cy/mm')
plt.plot(x[4700:6800], data_sampled[4700:6800] / np.sum(psf), label = 'Output: 2.5 cy/mm')
plt.plot(x[6800:10000], data_sampled[6800:10000] / np.sum(psf), label = 'Output: 1 cy/mm')

plt.xlim(x[0], x[-1])
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=11)
plt.xlabel('x [mm]')
plt.ylabel('Intensity [a.u.]')
plt.grid(True, which='major', linestyle='-')
plt.grid(True, which='minor', linestyle='--', alpha=0.4)

