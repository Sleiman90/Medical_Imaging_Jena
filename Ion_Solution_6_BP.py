#!/usr/bin/env python
# coding: utf-8

# Again, we start by importing numpy and matplotlib and by setting additional default plotting options. Additionally, we import the [loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) function from the SciPy package for loading the input data.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# nicer default plot settings
plt.rcParams.update({'font.size': 16, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'figure.figsize': (13,7)})


# We simply copy/paste the given implementation of the backprojection from the assignment sheet. The function takes two input parameters: ```sinogram``` is a 2D matrix which represents the sinogram of the projections with the first dimension containing the projection points and the second dimension the different rotation angles. The ```theta``` parameter is an array of the actual rotation angles under which the projections in the sinogram were acquired.

# In[3]:


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


# We start by loading the provided MATLAB file and then extract the data as specified in the assignment sheet. To see what kind of data we are dealing with, we also display the loaded sinogram.

# In[4]:


# load data
img = loadmat('../assignments/Ion_Assignment_6_FBPData.mat')

# extract the projections and angles
proj = np.array(img['proj'])
angles = np.squeeze(np.array(img['angles']))
print(proj.shape)
print(angles.shape)

# display sinogram
plt.imshow(proj, cmap = 'gray', aspect = 'auto')
ax = plt.gca()
ax.set_aspect(0.5)
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_xticklabels(angles[[0, 45, 90, 135, 180]])
ax.set_xlabel('Projection angle [deg]')
ax.set_ylabel('Projection point')
ax.yaxis.tick_right()


# To understand the concept of the backprojection, we will calculate the backprojected images by using only single projections as input. We do this separately for the projection angles of 0°, 45° and 65°. The result shows how the respective projection profile is effectively 'smeared' over the entire image under the projection's acquisition angle:

# In[5]:


fig, ax = plt.subplots(ncols = 2, nrows = 3, figsize=(11,11))

# angles used
plot_angles = np.array([0, 45, 65])

# loop over projection angles
for angle, angle_ind in zip(plot_angles, range(len(plot_angles))):
    
    # get the single projection for the given angle
    proj_angle = proj[:, angle]
    img_angle = backproject(proj_angle, angle)
        
    ax[angle_ind,0].plot(proj_angle)
    ax[angle_ind,1].imshow(img_angle, cmap = 'gray', vmin = 0, vmax = 180)
    
    ax[angle_ind,0].grid(True, which='major', linestyle='-')
    ax[angle_ind,0].grid(True, which='minor', linestyle='--', alpha=0.4)
    ax[angle_ind,0].set_xlim(0, len(proj))
    ax[angle_ind,0].set_title(f'Projection angle {angle}°', fontsize=16)
    ax[angle_ind,0].set_xlabel('Projection point')
    ax[angle_ind,0].set_ylabel('Projection intensity')
    ax[angle_ind,0].set_ylim(-10, 180)
    
fig.tight_layout() 


# Now let's create an image by backprojecting the three projections above - which is nothing more than the sum of the above images.

# In[6]:


img = backproject(proj[:, plot_angles], angles[plot_angles])
plt.imshow(img, cmap = 'gray')
plt.title('Backprojection from only 3 projections')


# Finally, we reconstruct the image by using all 181 projections. The result resembles a cross, but it is very blurred because we have only used an unfiltered back projection so far.

# In[7]:


img = backproject(proj, angles)
plt.imshow(img, cmap = 'gray')
plt.title('Backprojection')

