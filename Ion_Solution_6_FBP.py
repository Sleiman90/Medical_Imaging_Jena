#!/usr/bin/env python
# coding: utf-8

# Again, we start by importing numpy and matplotlib and by setting additional default plotting options. Additionally, we import the [loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) function from the SciPy package for loading the input data.

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# nicer default plot settings
plt.rcParams.update({'font.size': 16, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'figure.figsize': (13,7)})


# We simply copy/paste the given implementation of the backprojection from the assignment sheet. The function takes two input parameters: ```sinogram``` is a 2D matrix which represents the sinogram of the projections with the first dimension containing the projection points and the second dimension the different rotation angles. The ```theta``` parameter is an array of the actual rotation angles under which the projections in the sinogram were acquired.

# In[8]:


def backproject(sinogram, theta):
    """Backprojection function. 
    inputs:  sinogram - [n x m] 2D array, n = projection data, m = angles
             theta - array of length m with the rotation angles
    output: backprojArray - [n x n] backprojected 2-D numpy array"""
    imageLen = sinogram.shape[0]
    reconMatrix = np.zeros((imageLen, imageLen))
    
    # create coordinate system centered at (x,y = 0,0)
    x = np.arange(imageLen)-imageLen/2 
    y = x.copy()
    Y, X = np.meshgrid(x, y)    

    sinogram = np.squeeze(sinogram)
    
    theta = theta*np.pi/180
    if not np.isscalar(theta):
        theta = np.squeeze(theta) # remove singleton dimensions
    if np.isscalar(theta):
        theta = np.array([theta]) # make numpy array    
    numAngles = len(theta)    

    for n in range(numAngles):
        # determine rotated x-coordinate about origin in mesh grid form
        Xrot = X*np.sin(theta[n])-Y*np.cos(theta[n]) 
        
        # shift back to original image coordinates, round values to make indices
        XrotCor = np.round(Xrot+imageLen/2) 
        XrotCor = XrotCor.astype('int')
        
        projMatrix = np.zeros((imageLen, imageLen))
        
        # after rotating, exclude coordinates that exceed the size of the original
        m0, m1 = np.where((XrotCor >= 0) & (XrotCor <= (imageLen-1)))
        
        if np.ndim(sinogram) == 1:
            s = sinogram
        else:
            s = sinogram[:,n] #get projection
        projMatrix[m0, m1] = s[XrotCor[m0, m1]]  # backproject in-bounds data
        reconMatrix += projMatrix
         
    return reconMatrix


# We start by loading the provided MATLAB file and then extract the data as specified in the assignment sheet. 

# In[9]:


# load data
img = loadmat('../assignments/Ion_Assignment_6_FBPData.mat')

# extract the 'dsa_series' images
proj = np.array(img['proj'])
angles = np.squeeze(np.array(img['angles']))
print(proj.shape)
print(angles.shape)


# Next, we transform our projections into Fourier space, design our filter, multiply it with the projections in Fourier space and transform back. Please note that after any Fourier transformation the resulting data will be complex valued. For this reason, we continue to use only the real part of the filtered projections:

# In[10]:


# calculate the Fourier transform of the projections
proj_fft = np.fft.fft(proj, axis = 0)

# design the filter
filt = 1 - 2 * np.abs(np.arange(0, proj_fft.shape[0]) / proj_fft.shape[0] - 0.5)
filt = np.expand_dims(filt, axis = 1) # add a singleton dimension at the end
print(filt.shape)

# multiply filter with the projections in Fourier space
proj_fft_filt = proj_fft * filt

# use inverse Fourier transform to go back to having projections
proj_filt = np.fft.ifft(proj_fft_filt, axis = 0)

# take real part
proj_filt = np.real(proj_filt)


# When comparing the original projection with the filtered projection, one can observe that all edges are strongly over attenuated and now also contain negative values. This is important because without negative values, the reconstructed image intensity can only increase with each projection added to the final image after backprojecting it.

# In[9]:


# plot first projection and filter
fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize=(11,6))
ax[0].plot(np.abs(proj_fft[:,0]), 'b-')
ax[0].set_xlim(0, proj_fft.shape[0])
ax[0].set_xlabel('Array index')
ax[0].set_ylabel('|FFT(proj)|', color='blue')
ax[0].tick_params(axis='y', colors='blue')
sec_ax = ax[0].twinx() # create second axis
sec_ax.plot(filt, 'r-', label='abc')
sec_ax.set_ylabel('Ram-lak filter', color='red')
sec_ax.tick_params(axis='y', colors='red')
sec_ax.spines['right'].set_color('red')
sec_ax.spines['left'].set_color('blue')

# plot original and filtered projections
ax[1].plot(proj[:,0], label = 'Original')
# scale by a factor of 5 for better visualization
ax[1].plot(proj_filt[:,0] * 5, label = 'Filtered')
ax[1].set_xlim(0, proj_fft.shape[0])
ax[1].legend()

for cax in ax:
    cax.grid(True, which='major', linestyle='-')
    cax.grid(True, which='minor', linestyle='--', alpha=0.4)

fig.tight_layout() 


# Finally, we use the filtered projections in the backprojection to obtain a much sharper image:

# In[10]:


fix, ax = plt.subplots(nrows = 1, ncols = 2)

# unfiltered backprojection
img = backproject(proj, angles)
ax[0].imshow(img, cmap = 'gray')
ax[0].set_title('Backprojection')

img = backproject(proj_filt, angles)
ax[1].imshow(img, cmap = 'gray', vmin=0)
ax[1].set_title('Filtered Backprojection')

