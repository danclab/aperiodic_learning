import mne 
import itertools
import os
from mne import read_epochs
from mne.time_frequency import psd_welch
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
import pandas as pd
from fooof import FOOOF
from fooof import FOOOFGroup
import numpy as np
import scipy.stats
import json
import collections
import pprint

motor_perturb_coh=[]
subj_id=[101,124,134,145]
for subj_id in subj_id:
    cats=['low','med','high']
    Perturbs = {-30.0,30.0}
    perturb_list=[]
    for perturb in Perturbs:
        cat_list=[]
        for cat in cats:
            session_params=[]
            for session in range(3,9):
                df=pd.read_csv(r'C:\Users\ytaza\Desktop\export\sub-%d\sub-%d-00%d-beh.csv'   % (subj_id, subj_id, session))
                epochs=mne.read_epochs(r'C:\Users\ytaza\Desktop\export\sub-%d\sub-%d-00%d-motor-epo.fif'  % (subj_id, subj_id, session))
                epochs=epochs.pick_types(meg=True, ref_meg=False,misc=False)
                psds,freqs= mne.time_frequency.psd_welch(epochs,fmin=0,fmax=120)
                idx=df[(df['perturb_cat'] == perturb) & (df['coh_cat'] == cat)].index.values
                mean_psds=np.mean(psds[idx,:,:],axis=0)
                fg = FOOOFGroup()
                fg.fit(freqs, mean_psds)
                r2s = fg.get_params('r_squared')
                idxr2 = np.where(r2s<0.95)
                exp =fg.get_params('aperiodic_params', 'exponent')
                exp=np.array(exp)
                exp[idxr2]=np.nan
                session_params.append(exp)
            cat_list.append(session_params)   
        perturb_list.append(cat_list)
    motor_perturb_coh.append(perturb_list)
motor_perturb_coh=np.array(motor_perturb_coh)  

motor_zero_inter=[]
subj_id=[101,124,134,145]
for subj_id in subj_id:
    session_params=[]
    for session in range(3,9):     
        df=pd.read_csv(r'C:\Users\ytaza\Desktop\export\sub-%d\sub-%d-00%d-beh.csv'   % (subj_id, subj_id, session))
        epochs=mne.read_epochs(r'C:\Users\ytaza\Desktop\export\sub-%d\sub-%d-00%d-motor-epo.fif'  % (subj_id, subj_id, session))  
        epochs=epochs.pick_types(meg=True, ref_meg=False,misc=False)
        psds,freqs= mne.time_frequency.psd_welch(epochs,fmin=0,fmax=120)
        idx=df[(df['perturb_cat'] == 0) & (df['coh_cat'] == 'zero')].index.values
        mean_psds=np.mean(psds[idx,:,:],axis=0)
        fg = FOOOFGroup()
        fg.fit(freqs, mean_psds)
        exp = fg.get_params('aperiodic_params', 'exponent')
        r2s = fg.get_params('r_squared')
        idxr2 = np.where(r2s<0.95)
        exp =fg.get_params('aperiodic_params', 'exponent')
        exp=np.array(exp)
        exp[idxr2]=np.nan
        session_params.append(exp)   
    motor_zero_inter.append(session_params)
motor_zero_inter=np.array(motor_zero_inter)


mpcmean=np.nanmean(motor_perturb_coh,axis=(0,3))
mpczmean=np.nanmean(motor_zero_inter,axis=(0,1))

M30low=mpcmean[0,0,:]
M30med=mpcmean[0,1,:]
M30high=mpcmean[0,2,:]
P30low=mpcmean[1,0,:]
P30med=mpcmean[1,1,:]
P30high=mpcmean[1,2,:]
Nm30l_z=np.subtract(M30low,mpczmean)                    
Nm30m_z=np.subtract(M30med,mpczmean)        
Nm30h_z=np.subtract(M30high,mpczmean)        
Np30l_z=np.subtract(P30low,mpczmean)
Np30m_z=np.subtract(P30med,mpczmean)       
Np30h_z=np.subtract(P30high,mpczmean)
V=np.concatenate((Nm30l_z,Nm30m_z,Nm30h_z,Np30l_z,Np30m_z,Np30h_z))

fig,[[ax1,ax2,ax3],[ax4,ax5,ax6]]= plt.subplots(2,3, figsize=[15,15],facecolor="white", sharey=False, sharex=False)
fig.suptitle('Exoponent in motor epochs n=4 R2>=0.95',fontsize=16,y=1)

im,cm =plot_topomap(Nm30l_z,epochs.info,cmap="RdBu_r",show=False, axes=ax1,vmax=np.max(V),vmin=np.min(V))
ax1.set(title="M30LOW-BASELINE")

im,cm =plot_topomap(Nm30m_z, epochs.info, cmap="RdBu_r",show=False, axes=ax2,vmax=np.max(V),vmin=np.min(V))
ax2.set(title="M30MED-BASELINE")

im,cm =plot_topomap(Nm30h_z,epochs.info,cmap="RdBu_r",show=False, axes=ax3,vmax=np.max(V),vmin=np.min(V))
ax3.set(title="M30HIGH-BASELINE")

im,cm =plot_topomap(Np30l_z,epochs.info,cmap="RdBu_r",show=False, axes=ax4,vmax=np.max(V),vmin=np.min(V))
ax4.set(title="P30LOW-BASELINE")

im,cm =plot_topomap(Np30m_z, epochs.info, cmap="RdBu_r",show=False, axes=ax5,vmax=np.max(V),vmin=np.min(V))
ax5.set(title="P30MED-BASELINE")

im,cm =plot_topomap(Np30h_z,epochs.info,cmap="RdBu_r",show=False, axes=ax6,vmax=np.max(V),vmin=np.min(V))
ax6.set(title="P30HIGH-BASELINE")

ax_x_start = 0.94
ax_x_width = 0.04
ax_y_start = 0.1
ax_y_height = 0.9
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
clb.ax.set_title('slope')


visual_perturb_coh=[]
subj_id=[101,124,134,145]
for subj_id in subj_id:
    cats=['low','med','high']
    Perturbs = {-30.0,30.0}
    perturb_list=[]
    for perturb in Perturbs:
        cat_list=[]
        for cat in cats:
            session_params=[]
            for session in range(3,9):
                df=pd.read_csv(r'C:\Users\ytaza\Desktop\export\sub-%d\sub-%d-00%d-beh.csv'   % (subj_id, subj_id, session))
                epochs=mne.read_epochs(r'C:\Users\ytaza\Desktop\export\sub-%d\sub-%d-00%d-visual-epo.fif'  % (subj_id, subj_id, session))
                epochs=epochs.pick_types(meg=True, ref_meg=False,misc=False)
                psds,freqs= mne.time_frequency.psd_welch(epochs,fmin=0,fmax=120)
                idx=df[(df['perturb_cat'] == perturb) & (df['coh_cat'] == cat)].index.values
                mean_psds=np.mean(psds[idx,:,:],axis=0)
                fg.fit(freqs, mean_psds)
                r2s = fg.get_params('r_squared')
                idxr2 = np.where(r2s<0.95)
                exp =fg.get_params('aperiodic_params', 'exponent')
                exp=np.array(exp)
                exp[idxr2]=np.nan
                session_params.append(exp)
            cat_list.append(session_params)   
        perturb_list.append(cat_list)
    visual_perturb_coh.append(perturb_list)
visual_perturb_coh=np.array(visual_perturb_coh)   


visual_zero_inter=[]
subj_id=[101,124,134,145]
for subj_id in subj_id:
    session_params=[]
    for session in range(3,9):     
        df=pd.read_csv(r'C:\Users\ytaza\Desktop\export\sub-%d\sub-%d-00%d-beh.csv'   % (subj_id, subj_id, session))
        epochs=mne.read_epochs(r'C:\Users\ytaza\Desktop\export\sub-%d\sub-%d-00%d-visual-epo.fif'  % (subj_id, subj_id, session))  
        epochs=epochs.pick_types(meg=True, ref_meg=False,misc=False)
        psds,freqs= mne.time_frequency.psd_welch(epochs,fmin=0,fmax=120)
        idx=df[(df['perturb_cat'] == 0) & (df['coh_cat'] == 'zero')].index.values
        mean_psds=np.mean(psds[idx,:,:],axis=0)
        fg.fit(freqs, mean_psds)
        r2s = fg.get_params('r_squared')
        idxr2 = np.where(r2s<0.95)
        exp =fg.get_params('aperiodic_params', 'exponent')
        exp=np.array(exp)
        exp[idxr2]=np.nan
        session_params.append(exp)
    visual_zero_inter.append(session_params)
visual_zero_inter=np.array(visual_zero_inter)

meanVPC=np.nanmean(visual_perturb_coh,axis=(0,3))
meanVPCZ=np.nanmean(visual_zero_inter,axis=(0,1))

M30low=meanVPC[0,0,:]
M30med=meanVPC[0,1,:]
M30high=meanVPC[0,2,:]
P30low=meanVPC[1,0,:]
P30med=meanVPC[1,1,:]
P30high=meanVPC[1,2,:]
Vm30l_z=np.subtract(M30low,meanVPCZ)                    
Vm30m_z=np.subtract(M30med,meanVPCZ)        
Vm30h_z=np.subtract(M30high,meanVPCZ)        
Vp30l_z=np.subtract(P30low,meanVPCZ)
Vp30m_z=np.subtract(P30med,meanVPCZ)       
Vp30h_z=np.subtract(P30high,meanVPCZ)
V=np.concatenate((Vm30l_z,Vm30m_z,Vm30h_z,Vp30l_z,Vp30m_z,Vp30h_z))

fig,[[ax1,ax2,ax3],[ax4,ax5,ax6]]= plt.subplots(2,3, figsize=[15,15],facecolor="white", sharey=False, sharex=False)
fig.suptitle('Coherance perturbation ineteraction in visaul epochs n=4',fontsize=16,y=1)

im,cm =plot_topomap(Vm30l_z,epochs.info,cmap="RdBu_r",show=False, axes=ax1,vmax=np.max(V),vmin=np.min(V))
ax1.set(title="M30LOW-BASELINE")

im,cm =plot_topomap(Vm30m_z, epochs.info, cmap="RdBu_r",show=False, axes=ax2,vmax=np.max(V),vmin=np.min(V))
ax2.set(title="M30MED-BASELINE")

im,cm =plot_topomap(Vm30h_z,epochs.info,cmap="RdBu_r",show=False, axes=ax3,vmax=np.max(V),vmin=np.min(V))
ax3.set(title="M30HIGH-BASELINE")

im,cm =plot_topomap(Vp30l_z,epochs.info,cmap="RdBu_r",show=False, axes=ax4,vmax=np.max(V),vmin=np.min(V))
ax4.set(title="P30LOW-BASELINE")

im,cm =plot_topomap(Vp30m_z, epochs.info, cmap="RdBu_r",show=False, axes=ax5,vmax=np.max(V),vmin=np.min(V))
ax5.set(title="P30MED-BASELINE")

plot_topomap(Vp30h_z,epochs.info,cmap="RdBu_r",show=False, axes=ax6,vmax=np.max(V),vmin=np.min(V))
ax6.set(title="P30HIGH-BASELINE")
ax_x_start = 0.94
ax_x_width = 0.04
ax_y_start = 0.1
ax_y_height = 0.9
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
clb.ax.set_title('slope')
