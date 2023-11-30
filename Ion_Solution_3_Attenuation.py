#!/usr/bin/env python
# coding: utf-8

# As python3 libraries we need numpy and matplotlib for numerical calculations and plotting. We also set some nicer plot settings.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# nicer default plot settings
plt.rcParams.update({'font.size': 20, 
                     'lines.linewidth': 2, 
                     'xtick.minor.visible': True, 
                     'ytick.minor.visible': True, 
                     'axes.grid': True, 
                     'figure.figsize': (12,7)})


# The mass attenuation coefficients for different materials can be downloaded from the [XCOM: Photon Cross Sections Database](https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html) of the National Institute of Standards and Technology (NIST) of the U.S. Department of Commerce. After downloading the tabular data as space-separated text files, they can be loaded using the [numpy.loadtxt()](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html) function in Python. Since the text files contain header information, we must omit the first two lines:

# In[6]:


data_h2o = np.loadtxt('Ion_Solution_3_MassAttenuationH2O.txt', skiprows=2)
data_fe = np.loadtxt('Ion_Solution_3_MassAttenuationFe.txt', skiprows=2)
data_pb = np.loadtxt('Ion_Solution_3_MassAttenuationPb.txt', skiprows=2)


# If we check the shape of the loaded data we seem that we have a table or 2D array with eight columns and a different number of rows for the three loaded materials/elements.

# In[7]:


print(data_h2o.shape)
print(data_fe.shape)
print(data_pb.shape)


# The content of each column is described in the header of the downloaded data and may differ depending on which columns you have selected to export on the XCOM website. For this sample solutions, we decided to export the data for all interaction processes, giving us eight different columns.
# 
# | Column | Content |
# | --- | --- |
# | 0 | Photon Energy (in MeV) |
# | 1 | Coherent scattering (Compton scattering) |
# | 2 | Incoherent scattering (Compton absorption) |
# | 3 | Photoelectric absorption |
# | 4 | Nuclear pair production |
# | 5 | Electron pair production (Triplet production) |
# | 6 | Total with coherent scattering |
# | 7 | Total without coherent scatter |

# To plot the total mass attenuation coefficient, we can either plot the column with index 6 or calculate the sum of the values of the columns with index 1 to 5. Since the data for the three elements/compounds have a different number of rows, we have to use their individual photon energies given in column 0 as x-coordinates for plotting:

# In[8]:


# plot data
plt.plot(data_h2o[:,0],np.sum(data_h2o[:,1:6], axis=1), label='H2O')
plt.plot(data_fe[:,0],np.sum(data_fe[:,1:6], axis=1), label='Fe')
plt.plot(data_pb[:,0],np.sum(data_pb[:,1:6], axis=1), label='Pb')

# logarithmic scaling, labels and nicer plot
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Energy (MeV)')
plt.ylabel('Mass attenuation coefficient (cm²/g)')
plt.legend()
plt.xlim(data_h2o[0,0], 1e3)
plt.grid(True, which='minor', linestyle='--', alpha=0.4)

# draw energy range used for diagnostic x-ray imaging
y1, y2 = plt.ylim()
plt.vlines(0.02, y1, y2, color = 'red')
plt.vlines(0.15, y1, y2, color = 'red')
plt.text(0.055, 1.5e4, 'diagnostic \nx-ray', color = 'red', ha = 'center', va = 'top', fontsize = 14)
plt.ylim((y1,y2))


# For photon energies typically used for diagnostic X-ray imaging, lead has by far the highest total mass attenuation coefficient. Interestingly, for higher energies between 1 MeV and 10 MeV, the mass attenuation coefficients of all three elements/compounds are very similar. To investigate the reason for this, we continue to plot the mass attenuation coefficients of the individual interaction processes. For the plot we use different line styles and colors to better distinguish the elements/compounds and interaction processes:

# In[9]:


fig, axes = plt.subplots(ncols=1)

# photoelectric absorption
photo_absorp, = axes.plot(data_h2o[:,0],data_h2o[:,3], label='H2O', color='red', linestyle='-')
axes.plot(data_fe[:,0],data_fe[:,3], label='Fe', color='blue', linestyle='-')
axes.plot(data_pb[:,0],data_pb[:,3], label='Pb', color='green', linestyle='-')

# coherent scattering
coh_scatt, = axes.plot(data_h2o[:,0],data_h2o[:,1], color='red', linestyle='--')
axes.plot(data_fe[:,0],data_fe[:,1], color='blue', linestyle='--')
axes.plot(data_pb[:,0],data_pb[:,1], color='green', linestyle='--')

# incoherent absorption
inc_scatt, = axes.plot(data_h2o[:,0],data_h2o[:,2], color='red', linestyle=':')
axes.plot(data_fe[:,0],data_fe[:,2], color='blue', linestyle=':')
axes.plot(data_pb[:,0],data_pb[:,2], color='green', linestyle=':')

# pair production
pair_prod, = axes.plot(data_h2o[:,0],data_h2o[:,4], color='red', linestyle='-.')
axes.plot(data_fe[:,0],data_fe[:,4], color='blue', linestyle='-.')
axes.plot(data_pb[:,0],data_pb[:,4], color='green', linestyle='-.')

# logarithmic scaling, labels and nicer plot 
plt.yscale('log')

plt.xscale('log')
plt.xlabel('Energy (MeV)')
plt.ylabel('Mass attenuation coefficient(cm²/g)')
plt.xlim(data_h2o[0,0], 1e3)
plt.ylim(1e-4, 1e4)
plt.grid(True, which='minor', linestyle='--', alpha=0.4)

# add two legends
lg1 = plt.legend(loc='right')
plt.legend([photo_absorp, coh_scatt, inc_scatt, pair_prod],
           ['Photoelectric Absorption', 'Coherent Scattering', 'Incoherent Scattering', 'Pair Production'])
axes.add_artist(lg1)


# The reason why all three elements/compounds have comparable total mass attenuation coefficients for energies in the range between 1 MeV and 10 MeV is that in this energy range Compton effect (incoherent scattering) is the dominating process. As discussed in one of the previous lectures, the mass attenuation coefficient for incoherent scattering is independent of Z, decreases slowly with photon energy, and is directly proportional to the number of electrons per gram, which varies by only 20% from the lightest to the heaviest elements (except hydrogen).
# 
# Finally, let's determine the total mass attenuation coefficients $\mu_m$ for all three elements for energies of 20 keV and 150 keV and plot the attenuation of an incoming x-ray beam (with normalized intensity) through a 10 cm thick slab. For simplicity we look in column 0 of the individual data to find the energy closest to 0.02 keV and 150 keV, and use the total mass attenuation coefficient from column 6 for these energies. To calculate the intensity of an x-ray beam for any distance $x$ within an material 
# $$
# I(x) = I_0 \cdot e^{-\mu_m \cdot \rho \cdot x}
# $$
# we also need the density $\rho$ of the elements/compounds and the intensity of the incoming x-ray beam, which we set to $I_0=1$ for simplicity.

# In[10]:


# density for the materials
rho_pb = 11.342 # g/cm³
rho_fe = 7.874  # g/cm³
rho_h2o = 0.998 # g/cm³

# thickness of the object in cm
d = np.linspace(0,2,1000)


# For plotting we will be using subplots and a loop over the plots and energies:

# In[11]:


fig, axes = plt.subplots(ncols = 2, nrows = 1)
axes = axes.flatten()

energies = np.array([0.02, 0.15]) # in MeV
for axis, energy in zip(axes, energies):     
    # find the index within the input arrays that is closest
    # to the energy of interest (e.g. 0.02 MeV and 0.15 MeV)
    ind_pb = np.abs(data_pb[:,0] - energy).argmin()
    ind_fe = np.abs(data_fe[:,0] - energy).argmin()
    ind_h2o = np.abs(data_h2o[:,0] - energy).argmin()
    
    # get the total mass attenuation coefficient from that index
    mu_pb = data_pb[ind_pb,6]
    mu_fe = data_fe[ind_fe,6]
    mu_h2o = data_h2o[ind_h2o,6]
    
    # calculate intensity over the thick object
    I_pb = 1 * np.exp(-mu_pb * rho_pb * d)
    I_fe = 1 * np.exp(-mu_fe * rho_fe * d)
    I_h2o = 1 * np.exp(-mu_h2o * rho_h2o * d)

    # plot curves
    axis.plot(d, I_h2o, label='H2O')
    axis.plot(d, I_fe, label='Fe')
    axis.plot(d, I_pb, label='Pb')

    # format plot
    axis.grid(True, which='minor', linestyle='--', alpha=0.4)
    axis.set_xlabel('Distance [cm]')
    axis.set_ylabel('$I/I_0$')
    axis.set_xlim(d[0],d[-1])
    axis.set_title(f'{energy * 1000} keV')
    axis.legend()


# Low-energy x-rays with energies of 20 keV are easily absorbed by thin iron or lead aprons. For lead, also applies to X-rays of higher energy of 150 keV. Water, on the other hand, is a very poor absorber due to its low density and is not really suitable for absorbing X-rays unless it is used in unrealistically large quantities.
