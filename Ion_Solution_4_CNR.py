#!/usr/bin/env python
# coding: utf-8

# Again, we start by importing numpy and matplotlib and by setting additional default plotting options. Additionally, we import the [loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) function from the SciPy package for loading the input image. The other two imports of [PathEffects](https://matplotlib.org/3.1.1/tutorials/advanced/patheffects_guide.html) and [mpltoolkits.axes_Grid1](https://matplotlib.org/mpl_toolkits/axes_grid1/index.html) are only used for an easier and better visualization of the results:

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat

# nicer default plot settings
plt.rcParams.update({'font.size': 20, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'figure.figsize': (12,7)})


# First, we use [loadmat()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) to load the provided image (the data is provided as a .mat MATLAB file so that both students working with MATLAB and Python can easily lose the data) and then we visualize this as an image using [imshow()](https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.imshow.html):

# In[2]:


# load data and print the content of the MAT file, .mat files
# will be loaded as a dictionary with data stored as 'key = value' pairs
img = loadmat('../assignments/Ion_Assignment_4_CNRData.mat')
print(img.keys())

# load the 'img_noise' data and convert to numpy array
img = np.array(img['img_noise'])

# display noisy image
im = plt.imshow(img, cmap='gray')
plt.text(256, 50, 'Object A', color='white', ha = 'center', size=24)
plt.text(512, 50, 'Object B', color='white', ha = 'center', size=24)
plt.text(768, 50, 'Object C', color='white', ha = 'center', size=24)

# add a nice colorbar
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax)


# To get a better feeling for the severity of the noise, we visualize a 1D profile line through the center of the matrix. We can simply do this by making a plot of the central line. We see that Object A can not clearly be seen in the 1D profile due to the strong noise in the image (suprisingly the human eye can still see it in the 2D image above!).

# In[3]:


profile = img[int(img.shape[0] / 2),:]

plt.plot(profile, linewidth=2)
plt.xlim(0, len(profile)-1)
plt.setp(plt.gca().spines.values(), linewidth=2)
plt.minorticks_on()
plt.grid(True, which='major', linestyle='-')
plt.grid(True, which='minor', linestyle='--', alpha=0.4)
plt.xlabel('Pixel in x-direction')
plt.ylabel('Intensity')


# In the next step, we have to create masks for each object and the background that we can use as region of interests (ROIs) to determine the mean and standard deviations in these areas. There are various ways to define region of interests. 
# 
# Since we are dealing with circular objects, we also want to create cirular ROIs. To do this, we create a coordinate grid of x,y value pairs for each image voxel by using the [meshgrid()](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) function. After calculating the distance of all image voxels from the center of each object, we can use the [where()](https://numpy.org/doc/stable/reference/generated/numpy.where.html) function to select those voxels where the calculated distances are smaller than a desired radius. Alternatively, using rectangular ROIs created by linear/colon indexing would be a sufficient solution in this case as well. 

# In[4]:


# Create an empty zero filled mask with the size of the image
mask = np.zeros(img.shape)

# In a rectangular region at the border set the mask value to 1 (noise)
mask[5:mask.shape[0] - 5, 5:50] = 1;

# Create an x-y parameter grid that contains coordinates for all voxels
x,y = np.meshgrid(np.arange(0,img.shape[1]), np.arange(0,img.shape[0]))

# Calculate the distance of each voxel from the center of Object A
r = np.sqrt((x - 256)**2 + (y - 128)**2)
# Where the distance is smaller than 48 voxels set mask value to 2
mask = mask + np.where(r < 48, 2, 0)

# Calculate the distance of each voxel from the center of Object B
r = np.sqrt((x - 512)**2 + (y - 128)**2)
# Where the distance is smaller than 48 voxels set mask value to 3
mask = mask + np.where(r < 48, 3, 0)

# Calculate the distance of each voxel from the center of Object C
r = np.sqrt((x - 768)**2 + (y - 128)**2)
# Where the distance is smaller than 48 voxels set mask value to 4
mask = mask + np.where(r < 48, 4, 0)

# display result
im = plt.imshow(mask, cmap='gray')
for x,t in zip([256, 512, 768], ['A','B','C']):
    txt = plt.text(x, 128, f'ROI {t}', color='white', ha = 'center', size=18, va = 'center')    
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
txt = plt.text(28, 128, f'Noise ROI', color='white', ha = 'center', size=18, va = 'center', rotation=90)    
txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])    
    
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax)


# Our mask is designed such that each individual ROI has a specific intensity value (e.g., 1, 2, 3 or 4), so that we can easily select different ROIs from the mask by using logical indexing like ```img_noise[mask == 1]```. Finally, let's calculate the CNR for all three regions of interests:

# In[5]:


val_std_noise = np.std(img[mask == 1])
val_mean_noise = np.mean(img[mask == 1])
for ind_roi in [2,3,4]:
    val_cnr = (np.mean(img[mask == ind_roi]) - val_mean_noise) / val_std_noise    
    print(f'CNR of ROI {ind_roi - 1} is: {val_cnr}')

