# -*- coding: utf-8 -*-
"""
AnalyzeRSoftSims.py
Created on Fri Oct 10 14:01:36 2025

Visualizes data from RSoft BeamPROP monitors.

@author: Anusha
"""

### IMPORTS ###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import linecache

### READ PHASE APPLIED ###
filename_pre = '2x2MMI_TwoInput_scan_Wav_Phase'
data_dir = f'C:/Users/runner/ActivePIC/2x2MMI/{filename_pre}_work/results/'
filename_1 = data_dir + f'{filename_pre}_bp_mon_1_last.dat'
filename_2 = data_dir + f'{filename_pre}_bp_mon_2_last.dat'
filename_3 = data_dir + f'{filename_pre}_bp_mon_3_last.dat'
filename_4 = data_dir + f'{filename_pre}_bp_mon_4_last.dat'

### READ OUTPUT 1 ###
data_1 = np.genfromtxt(filename_1, delimiter='',skip_header=4)
wav_line = linecache.getline(filename_1, 3)
phase_line = linecache.getline(filename_1, 4)
wav_line_split = wav_line.split(' ')
phase_line_split = phase_line.split(' ')
wav = np.linspace(float(wav_line_split[1]), float(wav_line_split[2]), int(wav_line_split[0]))
phase = np.linspace(float(phase_line_split[1]), float(phase_line_split[2]), int(phase_line_split[0]))

### READ OUTPUT 2 ###
data_2 = np.genfromtxt(filename_2, delimiter='', skip_header=4)

### READ OUTPUT 3 ###
data_3 = np.genfromtxt(filename_3, delimiter='', skip_header=4)

### READ OUTPUT 4 ###
data_4 = np.genfromtxt(filename_4, delimiter='', skip_header=4)

### 2D PLOT ###

# Plot each output
vmin, vmax, fontsize = 0, 0.6, 18
fig, ax = plt.subplots(1, 3, figsize=(24,6))
cmap = cm.viridis
cmap.set_bad(color='gray') 
im = ax[0].pcolormesh(phase, wav, data_1, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1].pcolormesh(phase, wav, data_2, cmap=cmap, vmin=vmin, vmax=vmax)
im2 = ax[2].pcolormesh(phase, wav, data_2 - data_1, cmap='RdBu_r')
fig.supxlabel('Phase shift (degrees)', fontsize=16)
ax[0].set_ylabel('Wavelength (microns)', fontsize=16)
cbar = fig.colorbar(im, ax=(ax[0], ax[1]), aspect=20)
cbar.set_label('Normalized intensity', rotation=270, labelpad=20, fontsize=16)
cbar = fig.colorbar(im2, ax=(ax[2]), aspect=20)
cbar.set_label('Normalized intensity', rotation=270, labelpad=20, fontsize=16)
ax[0].set_title('Output A', fontsize=fontsize)
ax[1].set_title('Output B', fontsize=fontsize)
ax[2].set_title('B - A', fontsize=fontsize)
plt.show()

vmin, vmax = 0, 360
fig, ax = plt.subplots(1, 3, figsize=(24,6))
cmap = cm.hsv
cmap.set_bad(color='gray') 
im = ax[0].pcolormesh(phase, wav, data_3, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1].pcolormesh(phase, wav, data_4, cmap=cmap, vmin=vmin, vmax=vmax)
outputPhase_res = (data_4 - data_3)
outputPhase_res = np.remainder(outputPhase_res, 360)
outputPhase_res = np.where(outputPhase_res > 180, outputPhase_res - 360, outputPhase_res)
im2 = ax[2].pcolormesh(phase, wav, outputPhase_res, cmap='RdBu_r')
fig.supxlabel('Phase shift (degrees)', fontsize=16)
ax[0].set_ylabel('Wavelength (microns)', fontsize=16)
cbar = fig.colorbar(im, ax=(ax[0], ax[1]), aspect=20)
cbar.set_label('Output phase (degrees)', rotation=270, labelpad=20, fontsize=16)
cbar = fig.colorbar(im2, ax=(ax[2]), aspect=20)
cbar.set_label('Normalized intensity', rotation=270, labelpad=20, fontsize=16)
ax[0].set_title('Output A', fontsize=fontsize)
ax[1].set_title('Output B', fontsize=fontsize)
ax[2].set_title('B - A', fontsize=fontsize)
plt.show()

# Plot loss
loss = 1 - (data_1 + data_2)

vmin, vmax, fontsize = 0, 0.6, 18
fig, ax = plt.subplots(1, 1, figsize=(10,6))
cmap = cm.viridis
cmap.set_bad(color='gray') 
im = ax.pcolormesh(phase, wav, loss, cmap=cmap, vmin=loss.min(), vmax=loss.max())
ax.set_xlabel('Phase shift(degrees)', fontsize=16)
ax.set_ylabel('Wavelength (microns)', fontsize=16)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Fractional loss', rotation=270, labelpad=20, fontsize=16)
plt.show()

