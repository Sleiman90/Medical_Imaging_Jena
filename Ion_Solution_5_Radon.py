#!/usr/bin/env python
# coding: utf-8

# Again, we start by importing numpy and matplotlib and by setting additional default plotting options. Additionally, we import the [rotate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html) function from the SciPy package which is required for the Radon transform function to work.

# In[1]:


from scipy.ndimage import rotate
import numpy as np
import matplotlib.pyplot as plt

# nicer default plot settings
plt.rcParams.update({'font.size': 20, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'lines.linewidth': 2,
                     'figure.figsize': (12,7)})


# We simply copy/paste the given implementation of the Radon transform from the assignment sheet. The function takes two input parameters: ```image``` is a 2D matrix which represents the image from which the projection is to be calculated and ```angles``` is an array that specifies the rotation angles from which the projections are to be calculated. For each rotation angle, the function simply rotates the input image so that a summation along the first Cartesian matrix dimension effectively creates the projection at the correct rotation angle. If an array of rotation angles is passed the resulting projections will be centered and zero padded to the size of the longest possible projection (corresponding to the image diagonal).

# In[2]:


def radon(image, angles):
    """
    This function calculates the radon transform
    :param image: 2D input image
    :param angles: single rotation angle or array of rotation angles in degree
    :return: 2D array of the projections for all angles
    
    If a list of rotation angles is passed the resulting projections will
    be centered and zero padded to the length of the longest possible
    projection (the image diagonal)
    """
    
    # check if angles are a scalar or numpy array
    if np.isscalar(angles):
        angles = np.array([angles]) # make numpy array
        
    # check if we only need to calculate for a single angle
    if len(angles) == 1:
        rotation = rotate(image, angles[0], output='float64')
        return sum(rotation)
    else:    
        # max length of the projections, equal to the image diagonal
        max_len = int(np.sqrt(2) * len(image))

        # initialize output array with zeros
        R = np.zeros((max_len, len(angles)), dtype='float64')

        for phi,  ind in zip(angles, range(len(angles))):
            # create a rotated copy of the input image 
            rotation = rotate(image, phi, output='float64')

            # center the projection along the projection dimension
            # in the output matrix (depending on the rotaiton angle
            # the projections will have a different length)
            start_ind = int(max_len / 2 - len(rotation) / 2)
            R[range(start_ind, start_ind + len(rotation)), ind] = sum(rotation)

        return R


# We also copy/paste the object definition from the assignment sheet and visualize the result using [matplotlib.pyplot.imshow()](https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.imshow.html). Additionally, we define an array with the three projection angles of 0°, 45° and 90°. For clarification we also overlay the projection axis with the T-shaped object:

# In[3]:


# define and display object
obj = np.zeros((256,256))
obj[40:72, 40:216] = 1.0
obj[40:208, 112:144] = 1.0

plt.imshow(obj, cmap='gray')

# define the three rotation angles of the projections
angles = np.array([0, 45, 90]) # in degree

# for clarity let's also add the projection angles
max_len = int(np.sqrt(2) * len(obj))

# coordinates of a horizontal line as (x,y) value pairs
p1 = np.array([-max_len / 2, 0])
p2 = np.array([max_len / 2, 0])

line_colors = []
for phi, ind in zip(angles, range(len(angles))):
    # get rotation angle in radiants
    phi = np.deg2rad(phi)
    
    # 2D rotation matrix
    rot = np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]])
    
    # rotated coorinates
    p1_rot = np.matmul(rot, p1)
    p2_rot = np.matmul(rot, p2)
    
    # plot the rotated line
    line, = plt.plot(np.array([p1_rot[0], p2_rot[0]]) + len(obj) / 2,
                     np.array([p1_rot[1], p2_rot[1]]) + len(obj) / 2, 
                     label=f'{np.rad2deg(phi):.1f}°')
    
    # store the colors of the overlayed lines color the projections the same
    line_colors.append(line.get_color())

# because the lines can extend outside of the image (they have a length
# of the image diagonal) we manually set the plot bounds
plt.xlim(0, 255)
plt.ylim(255, 0)
plt.legend(loc='lower left')


# Finally, we calculate the projections of the T-shaped object under the given rotation angles and plot the results using subplots:

# In[5]:


# create subplots and loop over the angles
fig, ax = plt.subplots(ncols = len(angles), nrows = 1, figsize=(13,4))
for ind in range(len(angles)):
    # calculate projections
    proj = radon(obj, angles[ind])
        
    # plot
    ax[ind].plot(proj, color = line_colors[ind])
    ax[ind].grid(True, which='major', linestyle='-')
    ax[ind].grid(True, which='minor', linestyle='--', alpha=0.4)
    ax[ind].tick_params(labelsize=16)
    ax[ind].set_xlim(0, len(proj))
    ax[ind].set_title(f'{angles[ind]}°', fontsize=16)
    ax[ind].set_xlabel('Projection point')
ax[0].set_ylabel('Projection intensity')


# To calculate the sinogram, we create a linearily spaced vector of rotation angles, going from 0 to 180 degrees in 181 steps. The result can then be visualized as an image using [matplotlib.pyplot.imshow()](https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.imshow.html):

# In[5]:


# define rotation angles for a complete sinogram
angles = np.linspace(0, 180, 181)

# calculate projection angles
proj = radon(obj, angles)

# display result
fig, ax = plt.subplots(ncols = 2, nrows = 1)
ax[0].imshow(obj, cmap='gray')
ax[0].set_title('Object')
ax[1].imshow(proj, cmap='gray')
ax[1].set_title('Sinogram')
ax[1].set_aspect(0.5)
ax[1].set_xticks([0, 45, 90, 135, 180])
ax[1].set_xticklabels(angles[[0, 45, 90, 135, 180]])
ax[1].set_xlabel('Projection angle [deg]')
ax[1].set_ylabel('Projection point')
ax[1].yaxis.tick_right()


# The Projection Slice Theorem states, that a projection of an object at a given angle is identical to taking the 2D Fast-Fourier Transform of that object, extracting a line through the center of the 2D Fourier representation at the same angle, and then performing an inverse 1D Fast-Fourier Transform of that central line. For simplicity, we will use a rotation angle of 0° to facilitate the extraction of this line from Fourier space. To test this theorem, we are performing the steps as described in the assignment sheet:

# In[6]:


# (1): perform 2D FFT of the object and FFTShift the zero frequency to the center
obj_fft = np.fft.fftshift(np.fft.fft2(obj))

# (2): extract the central 
obj_fft_center = obj_fft[128, :]

# (3): 1D inverse FFT of the central line
obj_fft_center_ifft = np.fft.ifft(obj_fft_center)

# Calculate projection using radon transform for a rotation angle of 0°
proj = radon(obj, 0)
print(proj.shape)

# (4) Plot result
fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (13,5))
ax[0].imshow(np.abs(obj_fft), cmap='gray')
ax[0].set_title('2D Fourier representation', fontsize=16)
ax[0].plot([0, 255], [128, 128],color='orange',linewidth=0.5)
ax[0].tick_params(labelsize=14)
ax[1].plot(np.abs(obj_fft_center_ifft), label = 'Fourier Transform')
#ax[1].plot(proj[np.arange(int(len(proj) / 2 - len(obj_fft_center_ifft) / 2),int(len(proj) / 2 + len(obj_fft_center_ifft) / 2))],'--', label='Radon Transform')
ax[1].plot(proj,'--', label='Radon Transform')
ax[1].legend(fontsize=14)
ax[1].set_title('Projection', fontsize=14)
ax[1].grid(True, which='major', linestyle='-')
ax[1].grid(True, which='minor', linestyle='--', alpha=0.4)
ax[1].tick_params(labelsize=14)
ax[1].set_xlabel('Projection point')
ax[1].set_ylabel('Projection intensity')

