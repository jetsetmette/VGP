import os
import numpy as np
from scipy import signal

import pandas as pd
from scipy import stats

# import matplotlib.pyplot as plt
# import trompy as tp

def get_number_cells (s2p_folder):
    iscell = np.load(os.path.join(s2p_folder, 'iscell.npy'))
    return sum(iscell[:,0])


def load_s2p_data(s2p_folder):
    raw_F = np.load(os.path.join(s2p_folder, 'F.npy'))
    neu_F = np.load(os.path.join(s2p_folder, 'Fneu.npy'))
    iscell = np.load(os.path.join(s2p_folder, 'iscell.npy'))
    s2p_length=len(raw_F[0,:])
    return raw_F, neu_F, iscell, s2p_length

def get_frames(events_file,s2p_length):
    df = pd.read_csv(events_file)

    pump_frames = list(df[df['Item1'] == 'pump_on']['Item2.Item2'])
    pump_frames=[frame for frame in pump_frames if frame < s2p_length-100] #removes event if too close to the end

    licks_frames = list (df[df['Item1']== 'Lick']['Item2.Item2'])
    licks_frames=[frame for frame in licks_frames if frame < s2p_length-100] #removes event if too close to the end
    
    return pump_frames, licks_frames

def process_cell(cell_idx, raw_F, neu_F, filter=False):
    x = raw_F[cell_idx, :] - 0.7*(neu_F[cell_idx, :])
    x = (x - np.mean(x))/np.std(x)
    if filter:
        x = filter_cell(x)
    return x

def filter_cell(x):
    t = np.arange(0,len(x)/10,0.1)
    filt = signal.butter(4, 1, 'low', fs=10, output='sos')
    filtered = signal.sosfilt(filt, x)
    return filtered

def get_snips(f, event_frames, baseline_frames=50, total_frames=150):
    snips = []
    for frame in event_frames:
        snips.append(f [frame - baseline_frames : frame + (total_frames - baseline_frames)])

    return np.array(snips)

def get_responsive(snips, range1=range(30,50), range2=range(50,70)):
    pre = np.mean(snips[:, range1], axis=0)
    post = np.mean(snips[:, range2], axis=0)
    
    return stats.ttest_rel(pre, post)

def get_licks_per_trial(pump_frames, licks_frames, trial_end=100):

    licks_per_trial=[]
    for p in pump_frames: 
        temp=[]
        for l in licks_frames: 
            if l > p and l < p + trial_end: 
                temp.append(l)                
        licks_per_trial.append(temp)

    return licks_per_trial

def get_first_lick(licks_per_trial):
    
    first_licks=[]
    for trial in licks_per_trial:
        if len(trial) > 0: 
            first_licks.append(trial[0])

    return first_licks

def get_ncells_overlap(cond1, cond2):
    
    n_both = sum(np.logical_and(cond1, cond2))
    n_neither = sum(~np.logical_or(cond1, cond2))
    n_cond1 = int(sum(cond1) - n_both)
    n_cond2 = int(sum(cond2) - n_both)
    
#     print(n_cond1, n_cond2, n_both, n_neither)

    return (int(sum(cond1)), int(sum(cond2)), n_cond1, n_cond2, n_both, n_neither)

def make_responsive_df(pump_responsive, lick_responsive):
    pump_r = abs(pump_responsive)
    lick_r = abs(lick_responsive)

    # activated cells
    pump_a = pump_responsive == 1
    lick_a = lick_responsive == 1

    # inhibited cells
    pump_i = pump_responsive == -1
    lick_i = lick_responsive == -1
#     print(pump_r)
#     print(pump_a)
#     print(pump_i)

    return pd.DataFrame([
                    get_ncells_overlap(pump_r, lick_r),
                    get_ncells_overlap(pump_a, lick_a),
                    get_ncells_overlap(pump_i, lick_i)],
                    columns=['pump_all','lick_all',"pump_only", "lick_only", "both", "neither"],
                    index=["responsive", "activated", "inhibited"]).T

def assemble_data(s2p_folder,events_file,
                  animal="unnamed", diet="ns", solution="ns",
                  baseline_frames=50, total_frames=150):

    raw_F, neu_F, iscell, s2p_length = load_s2p_data(s2p_folder)
    
    pump_frames, licks_frames= get_frames(events_file,s2p_length)
    
    cell_idx = [idx for idx,vals in enumerate(iscell) if vals[0]==1]

    pump_snips_all = []
    lick_snips_all = []
    pump_responsive = np.zeros(len(cell_idx))
    lick_responsive = np.zeros(len(cell_idx))

    for i, cell in enumerate(cell_idx):
        delta_f = process_cell(cell, raw_F, neu_F)

        pump_snips = get_snips(delta_f, pump_frames, baseline_frames=baseline_frames, total_frames=total_frames)
        r, p = get_responsive(pump_snips)

        if (p < 0.05) & (r < 0):
            pump_responsive[i] = 1
        elif (p < 0.05) & (r > 0):
            pump_responsive[i] = -1

        licks_per_trial = get_licks_per_trial(pump_frames, licks_frames, trial_end=total_frames-baseline_frames)
        first_lick_frames = get_first_lick(licks_per_trial)

        lick_snips = get_snips(delta_f, first_lick_frames, baseline_frames=baseline_frames, total_frames=total_frames)
        r, p = get_responsive(lick_snips)

        if (p < 0.05) & (r < 0):
            lick_responsive[i] = 1
        elif (p < 0.05) & (r > 0):
            lick_responsive[i] = -1

        pump_snips_all.append(pump_snips)
        lick_snips_all.append(lick_snips)

    df_responsive = make_responsive_df(pump_responsive, lick_responsive)

    return {"animal": animal,
            "diet": diet,
            "solution": solution,
            "raw_F": raw_F,
            "neu_F": neu_F,
            "iscell": iscell,
            "pump_snips_all": np.array(pump_snips_all),
            "pump_responsive": pump_responsive,
            "lick_snips_all": np.array(lick_snips_all),
            "lick_responsive": lick_responsive,
            "df_responsive": df_responsive
            }

        

        

    
#     both_activated_cells=list(set(pump_activated_cells)& set(lick_activated_cells))
#     both_inhibited_cells=list(set(pump_inhibited_cells)& set(lick_inhibited_cells))
    
#     print('pump activated:',pump_activated_cells)
#     print('pump inhibited:',pump_inhibited_cells)
#     print('lick activated:',lick_activated_cells)
#     print('lick inhibited:',lick_inhibited_cells)
#     print('both activated:',both_activated_cells)
#     print('both inhibited:', both_inhibited_cells)
    
#     # Number of cells activated by pump or lick
    
#     n_pump_a_only=len(pump_activated_cells)-len(both_activated_cells)
#     n_lick_a_only=len(lick_activated_cells)-len(both_activated_cells)
#     n_both_a=len(both_activated_cells)
#     n_non_a=len(cell_idx)-n_pump_a_only-n_lick_a_only-n_both_a
    
#     # Number of cells inhibited by pump or lick
#     n_pump_i_only=len(pump_inhibited_cells)-len(both_inhibited_cells)
#     n_lick_i_only=len(lick_inhibited_cells)-len(both_inhibited_cells)
#     n_both_i=len(both_inhibited_cells)
#     n_non_i=len(cell_idx)-n_pump_i_only-n_lick_i_only-n_both_i
    
#     # Number of cells response to pump or lick 
#     n_pump_r= n_pump_a_only+n_pump_i_only
#     n_lick_r= n_lick_a_only+n_lick_i_only
#     n_both_r= n_both_a + n_both_i
#     n_non_r= len(cell_idx)-n_pump_a_only-n_pump_i_only-n_lick_a_only-n_lick_i_only-n_both_a-n_both_i
    
#     print(n_pump_a_only,n_lick_a_only,n_both_a,n_pump_i_only,n_lick_i_only,n_both_i,n_non_r)
#     print(len(cell_idx))
    
#     sizes_a=[n_pump_a_only,n_lick_a_only,n_both_a,n_non_a]
#     sizes_i= [n_pump_i_only,n_lick_i_only, n_both_i,n_non_i]
#     sizes_r=[n_pump_r,n_lick_r,n_both_r,n_non_r]
#     labels= 'Pump','Lick','Both','Neither'
    
#     # pie chart for activated cells 
#     f1, ax1 = plt.subplots()
#     ax1.pie(sizes_a, labels=labels, autopct='%1.1f%%',
#         startangle=90, colors=['green','red','yellow','dimgrey'])
    
#     #pie chart for inhibited cells
    
#     f2, ax1 = plt.subplots()
#     ax1.pie(sizes_i, labels=labels, autopct='%1.1f%%',
#         startangle=90, colors=['green','red','yellow','dimgrey'])
    
#     #pie chart for responsive cells 
    
#     f3, ax1 = plt.subplots()
#     ax1.pie(sizes_r, labels=labels, autopct='%1.1f%%',
#         startangle=90, colors=['green','red','yellow','dimgrey'])
    
#     return mean_lick_a_snips, mean_lick_i_snips,mean_pump_a_snips,mean_pump_i_snips
    
  
#         Snips and p-value hit trials        
                
#         hit_snips=[]
#         pre_hit=[]
#         post_hit=[]
        
#         for p in hit: 
#             hit_snips.append(x[p-50:p+100])
#             pre_hit.append(np.mean(x[p-50:p]))
#             post_hit.append(np.mean(x[p:p+50]))
#         hit_result= stats.ttest_rel(pre_hit, post_hit)
        
#         #Snips and p-value missed trials
        
#         miss_snips=[]
#         pre_miss=[]
#         post_miss=[]
        
#         for p in miss: 
#             miss_snips.append(x[p-50:p+100])
#             pre_miss.append(np.mean(x[p-50:p]))
#             post_miss.append(np.mean(x[p:p+50]))
#         miss_result= stats.ttest_rel(pre_miss, post_miss)
