#!/usr/bin/env python
# coding: utf-8

# Again, we start by importing numpy and matplotlib and by setting additional default plotting options. We also import the [loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) function from the SciPy package for loading the input image.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scp
from mpl_toolkits.axes_grid1 import make_axes_locatable

# nicer default plot settings
plt.rcParams.update({'font.size': 20, 
                     'lines.linewidth': 2, 
                     'figure.figsize': (12,7)})


# We start by loading the matlab file which contains the image and by extracting it as a variable:

# In[2]:


dataset = scp.loadmat('../assignments/Ion_Assignment_5_CTData.mat')
img_ct = np.array(dataset['img_ct'])


# Let's now look into the shape and scaling of the loaded image

# In[3]:


print(f'Shape of the image {img_ct.shape}')
print(f'Maximum intensity value: {np.amax(img_ct)}')
print(f'Mininum intensity value: {np.amin(img_ct)}')


# The loaded intensity values are not given as CT numbers or Hounsfield units, so we rescale the image using the slope and offset given in the assignment:

# In[4]:


slope = 1/5
offset = -1024
img_ct_scale = img_ct * slope + offset
print(f'Maximum intensity value: {np.max(img_ct_scale)}')
print(f'Mininum intensity value: {np.min(img_ct_scale)}')


# Now we can look at the CT image using the [np.imshow()](https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html) function and add a [np.colorbar()](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.colorbar.html) that corresponds to the CT number in each voxel:

# In[7]:


plt.imshow(img_ct_scale, cmap='gray')
plt.colorbar()


# Before we go into details about windowing and filtering, we want to take a look at the distribution of the CT numbers in the given image. This can be done by calculating the histogram of the image. To better visualize the histogram, we scale the y-axis to only 15% of the max value (otherwise the high number of air voxels would stretch out the y-axis substantially, making it harder to interpret the histogram):

# In[8]:


hist_val, hist_bins, hist_patches = plt.hist(img_ct_scale.flatten(), bins = 64, rwidth=0.9)
ax = plt.gca()
ylim = ax.get_ylim()
ax.set_ylim(0, ylim[1] * 0.15)
ax.set_xlabel('CT Number')
ax.set_ylabel('Number of voxels')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.4)
ax.tick_params(labelsize=20)

air_Bins = (hist_bins < -950);
soft_bins = (hist_bins > -250) * (hist_bins < 250);
bone_bins = (hist_bins > 250) * (hist_bins < 2500);
for p, b in zip(hist_patches, air_Bins):
    if b:
        p.set_facecolor('green')
for p, b in zip(hist_patches, soft_bins):
    if b:
        p.set_facecolor('red')
for p, b in zip(hist_patches, bone_bins):
    if b:
        p.set_facecolor('orange')
plt.legend((hist_patches[0], hist_patches[15], hist_patches[35], hist_patches[5]), ('Air','Soft tissues','Compact bone','Other tissues'))


# The histogram shows a distinct peak around the CT number of -1000 (green) and a broad peak around 0 (red). The first peak (green) corresponds to air and the latter (red) can be ascribed to soft tissues composed of water and fat. CT numbers of 250 and above (orange) can be ascribed to spongious as well as compact bone.
# 
# Next, we want to window the image differently by applying a bone window. A window is typically defined by specifying a window center and window width according to the image below. This effectively changes the mapping between the CT numbers and the displayed gray-scale values and thus increases the dynamic range of the displayed image (because more gray values can be used to display a smaller range of CT numbers).
# 
# <img src="../images/CT_Windowing.png" width=500px>
# 
# To implement the windowing, we only have to change how the data is displayed, i.e. change the min and max displayed gray scale values. We can do this by specifying the ```vmax``` and ```vmin``` parameters of the [imshow()](https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.imshow.html) function based on our defined window. The resulting image highlights only bony structures:

# In[12]:


window_center = 1000;
window_width = 1500;
plt.imshow(img_ct_scale, cmap='gray', vmin = window_center - window_width / 2, vmax = window_center + window_width / 2)
plt.colorbar()


# Now, we want not only window the image, but also apply a filter, according to the image below. Compared to a simple window, the filter sets values that are outside the upper bounds of the window range to zero (instead of setting them to the maximum).
# 
# <img src="../images/CT_Filtering.png" width=500px>
# 
# To implement the filter, we make a copy of the scaled CT image and, after creating variables for the filter center and filter width, use the [np.where()](https://numpy.org/doc/stable/reference/generated/numpy.where.html) function to filter the image. All values that are outside the filter range will be set to the lower bounds of the filter (i.e. ```filter_center - filter_width / 2```). The result is a soft tissue filtered image with all bone removed and features a strong soft tissue contrast:

# In[13]:


filter_center = 0
filter_width = 600
img_ct_soft = img_ct_scale
img_ct_soft = np.where(img_ct_soft < filter_center - filter_width / 2, filter_center - filter_width / 2, img_ct_soft)
img_ct_soft = np.where(img_ct_soft > filter_center + filter_width / 2, filter_center - filter_width / 2, img_ct_soft)
plt.imshow(img_ct_soft, cmap='gray')
plt.colorbar()


# Finally, let's show all images side by side for a better comparison:

# In[14]:


fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize=(13,5))

ax[0].axis('off')   
ax[0].set_title('Full window', fontsize=16)
ax[0].imshow(img_ct_scale, cmap='gray')

ax[1].imshow(img_ct_scale, cmap='gray', vmin = window_center - window_width / 2, vmax = window_center + window_width / 2)
ax[1].set_title('Bone window', fontsize=16)
ax[1].axis('off')

ax[2].imshow(img_ct_soft, cmap='gray')
ax[2].set_title('Soft tissue filter', fontsize=16)
ax[2].axis('off')

