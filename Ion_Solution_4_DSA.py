#!/usr/bin/env python
# coding: utf-8

# Again, we start by importing numpy and matplotlib and by setting additional default plotting options. Additionally, we import the [loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) function from the SciPy package for loading the input image. The other import of [mpl_toolkits.axes_grid1.ImageGrid](https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.axes_grid1.axes_grid.ImageGrid.html) is used for better grid-like visualization of multiple images.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.io import loadmat

# nicer default plot settings
plt.rcParams.update({'font.size': 16, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'figure.figsize': (13,7)})

# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore');


# As in the previous task, we use [loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) to load the provided images. Please note that we have to manually specify the type of the images as ```float32``` (otherwise NumPy tries to recognize the data type and sets it as ```uInt32```, which, although correct, would cause problems when calculating the difference images). When looking at the shape, we observe that the data has an additional third-dimension, basically representing a stack of images:

# In[2]:


# load data
img = loadmat('../assignments/Ion_Assignment_4_DSAData.mat')

# extract the 'dsa_series' images
img = np.array(img['dsa_series'], dtype='float32')
print(img.shape)


# To visualize the loaded fluoroscopy images, we use the [ImageGrid](https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.axes_grid1.axes_grid.ImageGrid.html) function to create a tight grid of subplots into which we can  put the images by using a loop. The images show 10 frames of a fluoroscopy scan of a human head, clearly showing the iodine-containing contrast agent flowing from the carotid artery into the brain.

# In[3]:


fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(2, 5), axes_pad=0.0)

for iFrame in np.arange(0, img.shape[2]):
    grid[iFrame].imshow(img[:,:,iFrame], cmap='gray')
    grid[iFrame].axis('off')


# To calculate the digital subtraction angiography (DSA) images, we take the logarithm of all images and substract the logarithmized first frame from them. This effectively substracts the static background and reveals only the vessels and shows the inflow of the contrast agent. Due to the logarithm operation, the resulting images can contains not-a-number (nan) values. So for better visualization, we select a rather narrow windowing between 0 and 0.5:

# In[4]:


img_diff = np.log(img[:,:,0].reshape((img.shape[0], img.shape[1], 1))) - np.log(img)

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(2, 5), axes_pad=0.0)

for iFrame in np.arange(0, img.shape[2]):
    grid[iFrame].imshow(img_diff[:,:,iFrame], cmap='gray', vmax = 0.5, vmin = 0)
    grid[iFrame].axis('off')    


# Calculating the Maximum Intensity Projection (MAP) over the calculated DSA images will pick the brightest signal value from all frames voxel by voxel and can thus be used to visualize the entire vessel network:

# In[5]:


map = np.amax(img_diff, axis=2)

plt.imshow(map, cmap='gray', vmin = 0, vmax = 0.5)
plt.title('Maximum intensity projection')


# To visualize the time course of the contrast agent flow, we simply select the two given voxels and plot the absolute values over the third dimension. The resulting plots nicely show the inflow of the contrast agent from the carotid artery at the bottom of the image where it reaches its peak earlier than in a smaller distal vessel.

# In[6]:


fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize=(11,8))

ax[0].imshow(map, cmap='gray', vmin = 0, vmax = 0.5)
ax[0].plot(440, 931, 'o')
ax[0].plot(145, 200, 'o')
ax[0].set_title('Maximum intensity projection')

ax[1].plot(np.abs(img_diff[931, 440, :]), label='Carotid Artery')
ax[1].plot(np.abs(img_diff[200, 145, :]), label='Distal Vessel')
ax[1].minorticks_on()
ax[1].grid(True, which='major', linestyle='-')
ax[1].grid(True, which='minor', linestyle='--', alpha=0.4)
ax[1].set_xlabel('Frame')
ax[1].set_ylabel('Intensity')
ax[1].yaxis.tick_right()
ax[1].set_xlim((0, img.shape[2] - 1))
ax[1].yaxis.set_label_position("right")
ax[1].set_title('Time course of contrast agent')
ax[1].legend()

