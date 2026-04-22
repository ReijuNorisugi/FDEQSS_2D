# -*- coding: utf-8 -*-
"""
Code for analyzing 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260422
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as mcolors
import os
import sys
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from pathlib import Path
sys.path.append(str(next(p for p in Path(__file__).resolve().parents if p.name == 'FDEQSS_2D')))
from sbiem_modules.utils import  utils

fname = '/RRF/test/'   # Please type directory name.

# Set the path to the host directory.
current = os.getcwd()
rel = ""

# Choose which to plot.
# When EQ_num is large, Nuclei is time-consuming.
Disp_V_cmap     = True # Plot displacement.
Check           = True # Check time steps and error.
Trajectory      = True # Plot trajectory.
Nuclei          = True # Plot evolution during dynamic events.
EQ_before_after = True # Plot shear stress and slip before and after events.
Summary         = True  # Plot source parameters.
Anime           = False # Plot animation.


yr2sec = 365.*24.*60.*60.
day2sec = 24.*60.*60.
hour2sec = 60.*60.
min2sec  = 60.

# Parameters to adjust style for displacement plot.
# Duration for coseismic snapshots.
duration = 0.1 #sec
# Duration for aseismic snapshots.
duration_ = 1.  # year
duration_ = duration_ * yr2sec

# The number of events you want to show in one displacement plot.
plot_num = 6
reduce = 1

downsample = 1    # Set relevant downsampling rate in space.

########################   Load parameters used in simulation   ########################s
Conditions = utils.load('{}Output{}Conditions.pkl'.format(rel, fname))
Medium =  utils.load('{}Output{}Medium.pkl'.format(rel, fname))
FaultParams =  utils.load('{}Output{}FaultParams.pkl'.format(rel, fname))

os.makedirs('{}Figures{}Disp_V_cmap'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Lapusta_plot'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/Tau_V'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/Tau_slip'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/V_state_1'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/V_state_2'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/V_state_4'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/V_state_8'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/V_state_11'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Nuclei/Velocity'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Nuclei/Tau'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Nuclei/State_1'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Nuclei/State_2'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Nuclei/State_4'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Nuclei/State_8'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Nuclei/State_11'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Nuclei/PSD_stack'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}EQ_before_after'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Check'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Summary'.format(rel, fname), exist_ok=True)


mirror = Conditions['mirror']
qd = Conditions['qd']
mode = Conditions['mode']
sparse = Conditions['sparse']
snap_EQ = Conditions['snap_EQ']
dense = Conditions['dense']
outerror = Conditions['outerror']
Frict = Conditions['Frict']
Stepper = Conditions['Stepper']
tmax = Conditions['tmax']
stations_uni = Conditions['stations_uni']
stations = Conditions['stations']


Cs = Medium['Cs']
Cp = Medium['Cp']
mu = Medium['mu']
nu = Medium['nu']
eta = Medium['eta']
V0 = Medium['V0']
Vpl = Medium['Vpl']
beta_min = Medium['beta_min']
lam = Medium['lam']
Nele = Medium['Nele']
hcell = Medium['hcell']
nperi = Medium['nperi']
kn = Medium['kn']
xp = Medium['xp']/1000.
xp = xp[::downsample]
ncell = Medium['ncell']
dtmin = Medium['dtmin']
Tw = Medium['Tw']

Lf = FaultParams['Lf']
D_inv = FaultParams['D_inv']
n_up = FaultParams['n_up']
n_low = FaultParams['n_low']
k_size = FaultParams['k_size']
n_array = FaultParams['n_array']
dim = FaultParams['dim']
Y_bar = FaultParams['Ybar']
Dc = FaultParams['Dc']
alpha = FaultParams['alpha']
beta = FaultParams['beta']
k = FaultParams['k']
a = FaultParams['a']
c = FaultParams['c']
f0 = FaultParams['f0']
sigma = FaultParams['sigma']
tau_ini = FaultParams['tau_ini']
tauc = FaultParams['tauc']
v_cos = np.min(2. * Cs * a * sigma / mu)
##############################################################

##########   Load counting for file open   #############
id_count = np.load('{}Output{}id_count.npy'.format(rel, fname))
id_count_st = np.load('{}Output{}id_count_st.npy'.format(rel, fname))
EQ_num = np.load('{}Output{}EQ_num.npy'.format(rel, fname))
id_station = np.load('{}Output{}id_station.npy'.format(rel, fname))
t_prep = np.load('{}Output{}t_prep.npy'.format(rel, fname))
t_wh = np.load('{}Output{}t_wh.npy'.format(rel, fname))
########################################################


###############################     Unpack binary outputs    ##################################
out_t_sparse   = np.fromfile('{}Output{}out_t_sparse.bin'.format(rel, fname), dtype=np.float64)
out_t_dense    = np.fromfile('{}Output{}out_t_dense.bin'.format(rel, fname), dtype=np.float64)

out_delta      = np.fromfile('{}Output{}out_delta.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_V          = np.fromfile('{}Output{}out_V.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_tau        = np.fromfile('{}Output{}out_tau.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_state_1    = np.fromfile('{}Output{}out_state_1.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_state_2    = np.fromfile('{}Output{}out_state_2.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_state_4    = np.fromfile('{}Output{}out_state_4.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_state_8    = np.fromfile('{}Output{}out_state_8.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_state_11   = np.fromfile('{}Output{}out_state_11.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_Gcd        = np.fromfile('{}Output{}out_Gcd.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))

out_Pr         = np.fromfile('{}Output{}out_Pr.bin'.format(rel, fname), dtype=np.float64)

EQcatalog      = np.fromfile('{}Output{}out_catalog.bin'.format(rel, fname), dtype=np.float64)
err_source     = np.fromfile('{}Output{}out_err_source.bin'.format(rel, fname), dtype=np.float64)
id_main        = np.fromfile('{}Output{}out_idmain.bin'.format(rel, fname), dtype=np.int64)


out_error      = np.fromfile('{}Output{}out_error.bin'.format(rel, fname), dtype=np.float64)
out_dt         = np.fromfile('{}Output{}out_dt.bin'.format(rel, fname), dtype=np.float64)

out_delta_st     = np.fromfile('{}Output{}out_delta_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
out_V_st         = np.fromfile('{}Output{}out_V_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
out_tau_st       = np.fromfile('{}Output{}out_tau_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
out_state1_st    = np.fromfile('{}Output{}out_state1_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
out_state2_st    = np.fromfile('{}Output{}out_state2_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
out_state4_st    = np.fromfile('{}Output{}out_state4_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
out_state8_st    = np.fromfile('{}Output{}out_state8_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
out_state11_st   = np.fromfile('{}Output{}out_state11_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
#########################################################################################################################


print('')
print('Preparation time {} sec'.format(t_prep))
if t_wh/day2sec >= 1.:
    print('Consuming time {} day'.format(t_wh/day2sec))
elif t_wh/hour2sec >= 1.:
    print('Consuming time {} hour'.format(t_wh/hour2sec))
elif t_wh/min2sec >= 5.:
    print('Consuming time {} min'.format(t_wh/min2sec))
else:
    print('Consuming time {} sec'.format(t_wh))
print('{} time steps'.format(id_count_st))
print('{} quakes'.format(EQ_num))
print('')


log_V = np.log10(out_V/Vpl)
thres = np.log10(v_cos/Vpl)
norm_V = mpl.colors.Normalize(vmin=0, vmax=np.max(log_V))

colors = [(0, '#b3b3b3'), (norm_V(thres)*1/5, '#47585c'), 
          (norm_V(thres)*2/5, '#3d4b99'), (norm_V(thres)*3/5, '#00a1e9'), (norm_V(thres)*4/5, '#e8e8e8'), 
          (norm_V(thres), '#ffccbb'), ((norm_V(thres)+1)*0.5, '#ff7f40'), (1, '#c70000')]
french_flag = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

fig, cbar = plt.subplots(figsize=(5, 0.5))
mpl.colorbar.Colorbar(
ax=cbar,
mappable=cm.ScalarMappable(norm=norm_V, cmap=french_flag),
orientation='horizontal',
).set_label(r'$\log_{10}(V/V_{\rm{pl}})$', fontsize=15)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fontsize=15)
plt.savefig('{}Figures{}Disp_V_cmap/colormap_v.png'.format(rel, fname), dpi=600, bbox_inches='tight')
plt.close()

if Disp_V_cmap:
    if EQ_num == 0:
        plot_num = 1#np.max(out_delta)                 # Displacement included in a single figure.
        plot_range = int(np.max(out_delta)//plot_num)  # Number of figures.
    else:
        plot_range = round(EQ_num//plot_num)           # Number of figures.

    # Plot color map of slip rate with displacement snapshots and Lapusta plot.
    min_delta = np.min(out_delta, axis=1)
    max_delta = np.max(out_delta, axis=1)
    for j in range(plot_range):
        if not sparse:
            break
        else:
            if EQ_num == 0:
                id_begin = np.argmin(np.abs(min_delta - plot_num*j))
                id_end = np.argmin(np.abs(max_delta - plot_num*(j+1)))
                begin = id_begin
                end = id_end
            else:
                if EQ_num == 1:
                    id_begin = id_main[:, 0]
                    id_end = id_main[:, 1]
                    begin = id_begin[0]
                    end = len(out_delta)-1
                else:
                    id_begin = id_main[plot_num*j:plot_num*(j+1), 0]
                    id_end = id_main[plot_num*j:plot_num*(j+1), 1]
                    begin = id_begin[0]
                    end = id_end[-1]
        
            xp_array = np.zeros_like(out_delta[begin:end, ::reduce])
            xp_array = xp[::reduce] + xp_array
            xp_array = xp_array.T
    
            color = np.log10(out_V[begin:end, ::reduce]/Vpl)
            color[color <= 0] = 0.
            color = color.T
            color = np.ravel(color)
            
            if mirror:
                figsize = (26, 11)
                
                points = np.vstack([out_delta[begin:end, ::reduce].T.ravel(), xp_array.ravel()]).T
                tri = Delaunay(points)
                triangulation = Triangulation(points[:, 0], points[:, 1], tri.simplices)
            
                fig, ax = plt.subplots(figsize=figsize)
                plt.tricontourf(triangulation, color, cmap=french_flag, levels=100, vmin=0, vmax=np.max(log_V))
            elif mode == 'IV':
                figsize = (26, 11)
                
                points = np.vstack([xp_array.ravel(), out_delta[begin:end, ::reduce].T.ravel()]).T
                tri = Delaunay(points)
                triangulation = Triangulation(points[:, 0], points[:, 1], tri.simplices)
                
                fig, ax = plt.subplots(figsize=figsize)
                plt.tricontourf(triangulation, color, cmap=french_flag, levels=100, vmin=0, vmax=np.max(log_V))
            else:
                figsize = (11, 26)
                
                points = np.vstack([xp_array.ravel(), out_delta[begin:end, ::reduce].T.ravel()]).T
                tri = Delaunay(points)
                triangulation = Triangulation(points[:, 0], points[:, 1], tri.simplices)
                
                fig, ax = plt.subplots(figsize=figsize)
                plt.tricontourf(triangulation, color, cmap=french_flag, levels=100, vmin=0, vmax=np.max(log_V))
                
        
            if EQ_num != 0:
                id_cos = np.empty(0, dtype=np.int64)
                for l in range(plot_num):
                    for i in range(100):
                        if out_t_sparse[id_begin[l]] + duration*i >= out_t_sparse[id_end[l]]:
                            break
                        else:
                            id_cos = np.append( id_cos, np.argmin( np.abs(out_t_sparse - out_t_sparse[id_begin[l]]-duration*i) ) )
        
            id_ass = np.empty(0, dtype=np.int64)
            id_ass = np.append(id_ass, 0)
            for i in range(100):
                if out_t_sparse[begin]+duration_*i <= out_t_sparse[end]:
                    id_ass = np.append(id_ass, np.argmin( np.abs(out_t_sparse - out_t_sparse[begin] - duration_*i)))
        
        
            if mirror:
                if EQ_num != 0:
                    ax.plot(out_delta[id_cos].T, xp, color='black', linestyle='dashed', lw=4, label='Every {:.2}s'.format(duration))
                    ax.plot(out_delta[id_end].T, xp, color='black', linestyle='solid', lw=6, label='Total slip')
                    ax.plot(out_delta[id_ass].T, xp, color='black', linestyle='solid', lw=4, label='Every {:.2}yr'.format(duration_/yr2sec))
                else:
                    ax.plot(out_delta[id_ass].T, xp, color='black', linestyle='solid', lw=4, label='Every {:.2}yr'.format(duration_/yr2sec))
                ax.set_xlabel('Displacement (m)', fontsize=35)
                ax.set_ylabel('Distance along fault (km)', fontsize=35)
                ax.set_xlim(np.min(out_delta[begin:end]), np.max(out_delta[begin:end]))
                ax.set_ylim(np.min(xp), np.max(xp))
                ax.invert_yaxis()
                plt.xticks(fontsize=30)
                plt.yticks([0, np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=30)
            else:
                if EQ_num != 0:
                    ax.plot(xp, out_delta[id_cos].T, color='black', linestyle='dashed', lw=4, label='Every {:.2}s'.format(duration))
                    ax.plot(xp, out_delta[id_end].T, color='black', linestyle='solid', lw=6, label='Total slip')
                    ax.plot(xp, out_delta[id_ass].T, color='black', linestyle='solid', lw=4, label='Every {:.2}yr'.format(duration_/yr2sec))
                else:
                    ax.plot(xp, out_delta[id_ass].T, color='black', linestyle='solid', lw=4, label='Every {:.2}yr'.format(duration_/yr2sec))
                ax.set_xlabel('Distance along fault (km)', fontsize=35)
                ax.set_ylabel('Displacement (m)', fontsize=35)
                ax.set_xlim(np.min(xp), np.max(xp))
                ax.set_ylim(np.min(out_delta[begin:end]), np.max(out_delta[begin:end]))
                plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                            np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=30)
                plt.yticks(fontsize=30)
            handles, labels = plt.gca().get_legend_handles_labels()
            if EQ_num == 0:
                plt.legend([handles[0]], [labels[0]],
                           loc='lower right', fontsize=30, framealpha=1)
            else:
                plt.legend([handles[0], handles[-1]], [labels[0], labels[-1]],
                           loc='lower right', fontsize=30, framealpha=1)
            plt.tight_layout()
            plt.savefig('{}Figures{}Disp_V_cmap/Displacement_{}.png'.format(rel, fname, j), dpi=600)
            plt.close()
        
        
            fig, ax = plt.subplots(figsize=figsize)
            if mirror:
                if EQ_num != 0:
                    ax.plot(out_delta[id_cos].T, xp, color='tomato', linestyle='solid', lw=4, label='Every {:.2}s'.format(duration))
                    ax.plot(out_delta[id_ass].T, xp, color='royalblue', linestyle='solid', lw=4, label='Every {:.2}yr'.format(duration_/yr2sec))
                    ax.plot(out_delta[id_end].T, xp, color='black', linestyle='solid', lw=6, label='Total slip')
                else:
                    ax.plot(out_delta[id_ass].T, xp, color='blue', linestyle='solid', lw=4, label='Every {:.2}yr'.format(duration_/yr2sec))
                ax.set_xlim(np.min(out_delta[begin:end]), np.max(out_delta[begin:end]))
                ax.set_ylim(np.min(xp), np.max(xp))
                ax.set_xlabel('Displacement (m)', fontsize=25)
                ax.set_ylabel('Distance along strike (km)', fontsize=25)
                ax.invert_yaxis()
                plt.xticks(fontsize=30)
                plt.yticks([0, np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=30)
            else:
                if EQ_num != 0:
                    ax.plot(xp, out_delta[id_cos].T, color='tomato', linestyle='solid', lw=4, label='Every {:.2}s'.format(duration))
                    ax.plot(xp, out_delta[id_ass].T, color='royalblue', linestyle='solid', lw=4, label='Every {:.2}yr'.format(duration_/yr2sec))
                    ax.plot(xp, out_delta[id_end].T, color='black', linestyle='solid', lw=6, label='Total slip')
                else:
                    ax.plot(xp, out_delta[id_ass].T, color='blue', linestyle='solid', lw=4, label='Every {:.2}yr'.format(duration_/yr2sec))
                ax.set_xlim(np.min(xp), np.max(xp))
                ax.set_ylim(np.min(out_delta[begin:end]), np.max(out_delta[begin:end]))
                ax.set_xlabel('Distance along strike (km)', fontsize=25)
                ax.set_ylabel('Displacement (m)', fontsize=25)
                plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                            np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=30)
                plt.yticks(fontsize=30)
            handles, labels = plt.gca().get_legend_handles_labels()
            if EQ_num == 0:
                plt.legend([handles[0]], [labels[0]],
                           loc='lower right', fontsize=30, framealpha=1)
            else:
                plt.legend([handles[0], handles[-plot_num-1]], [labels[0], labels[-plot_num-1]],
                       loc='lower right', fontsize=30, framealpha=1)
            plt.tight_layout()
            plt.savefig('{}Figures{}Lapusta_plot/Displacement_{}.png'.format(rel, fname, j), dpi=600)
            if j == 2:
                plt.savefig('{}Figures{}Lapusta_plot/Displacement_{}.eps'.format(rel, fname, j))
            
            plt.close()


if Check:
    if outerror:
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(out_t_dense/yr2sec, out_error, color='black', lw=3)
        ax.set_xlabel('Time (year)', fontsize=15)
        ax.set_ylabel(r'Error $\epsilon$', fontsize=15)
        ax.set_yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig('{}Figures{}Check/Error.png'.format(rel, fname), dpi=600)
        
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(out_t_dense/yr2sec, out_dt, color='black', lw=3)
        ax.set_xlabel('Time (year)', fontsize=15)
        ax.set_ylabel(r'Time step $dt$ (sec)', fontsize=15)
        ax.set_yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig('{}Figures{}Check/Time_steps.png'.format(rel, fname), dpi=600)
        
        plt.close()
  
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(out_t_dense/yr2sec, out_error/out_dt, color='black', lw=3)
        ax.set_xlabel('Time (year)', fontsize=15)
        ax.set_ylabel(r'Ratio $\epsilon/dt$ (/sec)', fontsize=15)
        ax.set_yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig('{}Figures{}Check/Ratio.png'.format(rel, fname), dpi=600)
        
        plt.close()


if Trajectory:
    if dense:
        plt.rcParams['xtick.direction'] = 'inout'
        plt.rcParams['ytick.direction'] = 'inout'
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['xtick.minor.size'] = 5
        plt.rcParams['ytick.minor.size'] = 5
        
        V_plot = np.linspace(-22, 2, 100)
        tau_ss = np.zeros((len(id_station), len(V_plot)), dtype=np.float64)
        Phi = np.zeros((len(id_station), len(V_plot)), dtype=np.float64)
        for i in range(len(id_station)):
            for j in range(len(V_plot)):
                Phi[i][j] = np.sum( (k**2)*((( beta[id_station[i]]*k*Y_bar )/( alpha[id_station[i]]*(10**V_plot[j]) + beta[id_station[i]]*k ))**2) )
                tau_ss[i][j] = ( tauc[id_station[i]] + a[id_station[i]]*sigma[id_station[i]]*np.log((10**V_plot[j])/V0) 
                                 + c[id_station[i]]*sigma[id_station[i]]*np.sqrt(Phi[i][j]) )
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(V_plot, tau_ss[stat, :]*(10**(-6)), color='darkorange', lw=3, label=r'$\tau_{\rm{ss}}$')
            ax.plot(np.log10(out_V_st[:, stat]), out_tau_st[:, stat]*(10**(-6)), color='black', lw=3, label='Trajectory')
            ax.set_xlabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=13)
            ax.set_ylabel('Shear stress (MPa)', fontsize=13)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.minorticks_on()
            plt.legend(loc='upper left', fontsize=13, framealpha=1)
            plt.grid()
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/Tau_V/TV_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
  
        print('Trajectory TV')
        
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 4))
            for EQ in range(EQ_num):
                x = out_delta_st[id_main[EQ, 2]:id_main[EQ, 3], stat] - out_delta_st[id_main[EQ, 2], stat]
                x = x * 10**(2)
                y = out_tau_st[id_main[EQ, 2]:id_main[EQ, 3], stat]*(10**(-6))
                ax.plot(x, y, color='black', lw=3)
            
            ax.set_xlabel('Coseismic slip (cm)', fontsize=15)
            ax.set_ylabel('Shear stress (MPa)', fontsize=15)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.grid()
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/Tau_slip/TS_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
            
        print('Trajectory TS')
        
        
        state_ss = np.zeros((len(id_station), len(V_plot)), dtype=np.float64)
        sn = 0
        for i in range(len(id_station)):
            state_ss[i, :] = beta[id_station[i]] * k[sn] * Y_bar[sn] / (alpha[id_station[i]] * (10**V_plot) + beta[id_station[i]] * k[sn])
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(V_plot, np.log10(state_ss[stat, :]),
                    color='darkorange', lw=3, label=r'$\theta_{\rm{ss}}$')
            ax.plot(np.log10(out_V_st[:, stat]), np.log10(out_state1_st[:, stat]),
                    color='black', lw=3, label='Trajectory')
            ax.set_xlabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{1})|)$ (m)', fontsize=15)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.minorticks_on()
            plt.legend(loc='lower left', fontsize=13, framealpha=1)
            plt.grid()
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/V_state_1/VY1_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
            
        print('Trajectory VY1')
        
        
        state_ss = np.zeros((len(id_station), len(V_plot)), dtype=np.float64)
        sn = 1
        for i in range(len(id_station)):
            state_ss[i, :] = beta[id_station[i]] * k[sn] * Y_bar[sn] / (alpha[id_station[i]] * (10**V_plot) + beta[id_station[i]] * k[sn])
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(V_plot, np.log10(state_ss[stat, :]),
                    color='darkorange', lw=3, label=r'$\theta_{\rm{ss}}$')
            ax.plot(np.log10(out_V_st[:, stat]), np.log10(out_state2_st[:, stat]),
                    color='black', lw=3, label='Trajectory')
            ax.set_xlabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{4})|)$ (m)', fontsize=15)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.minorticks_on()
            plt.legend(loc='lower left', fontsize=13, framealpha=1)
            plt.grid()
            
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/V_state_2/VY2_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
            
        print('Trajectory VY2')
        
        
        state_ss = np.zeros((len(id_station), len(V_plot)), dtype=np.float64)
        sn = 15
        for i in range(len(id_station)):
            state_ss[i, :] = beta[id_station[i]] * k[sn] * Y_bar[sn] / (alpha[id_station[i]] * (10**V_plot) + beta[id_station[i]] * k[sn])
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(V_plot, np.log10(state_ss[stat, :]),
                    color='darkorange', lw=3, label=r'$\theta_{\rm{ss}}$')
            ax.plot(np.log10(out_V_st[:, stat]), np.log10(out_state4_st[:, stat]),
                    color='black', lw=3, label='Trajectory')
            ax.set_xlabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{16})|)$ (m)', fontsize=15)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.minorticks_on()
            plt.legend(loc='lower left', fontsize=13, framealpha=1)
            plt.grid()
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/V_state_4/VY4_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
            
        print('Trajectory VY4')
        
        
        state_ss = np.zeros((len(id_station), len(V_plot)), dtype=np.float64)
        sn = 255
        for i in range(len(id_station)):
            state_ss[i, :] = beta[id_station[i]] * k[sn] * Y_bar[sn] / (alpha[id_station[i]] * (10**V_plot) + beta[id_station[i]] * k[sn])        
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(V_plot, np.log10(state_ss[stat, :]),
                    color='darkorange', lw=3, label=r'$\theta_{\rm{ss}}$')
            ax.plot(np.log10(out_V_st[:, stat]), np.log10(out_state8_st[:, stat]),
                    color='black', lw=3, label='Trajectory')
            ax.set_xlabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{256})|)$ (m)', fontsize=15)
            
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.minorticks_on()
            plt.legend(loc='lower left', fontsize=13, framealpha=1)
            plt.grid()
            
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/V_state_8/VY8_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
            
        print('Trajectory VY8')
        
        
        state_ss = np.zeros((len(id_station), len(V_plot)), dtype=np.float64)
        sn = -1
        for i in range(len(id_station)):
            state_ss[i, :] = beta[id_station[i]] * k[sn] * Y_bar[sn] / (alpha[id_station[i]] * (10**V_plot) + beta[id_station[i]] * k[sn])
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(V_plot, np.log10(state_ss[stat, :]),
                    color='darkorange', lw=3, label=r'$\theta_{\rm{ss}}$')
            ax.plot(np.log10(out_V_st[:, stat]), np.log10(out_state11_st[:, stat]),
                    color='black', lw=3, label='Trajectory')
            ax.set_xlabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{2048})|)$ (m)', fontsize=15)
            
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.minorticks_on()
            plt.legend(loc='lower left', fontsize=13, framealpha=1)
            plt.grid()
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/V_state_11/VY11_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
            
        print('Trajectory VY11')


if EQ_num == 0:
    sys.exit()
else:
    pass


if Nuclei:
    if sparse:
        plt.rcParams['xtick.direction'] = 'inout'
        plt.rcParams['ytick.direction'] = 'inout'
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['xtick.minor.size'] = 5
        plt.rcParams['ytick.minor.size'] = 5
        
        skip = 10
        skip_ = 1
        tips = np.empty(0, dtype=np.int64)
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(3000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 2*Vpl:
                    length_ = i
                    break
            
            length_ = round(length_/skip_)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, np.log10(out_V[int(id_main[EQ, 0])-skip_*i, :]).T, color='black', lw=1.)
                if EQ == 10:
                    tips = np.append(tips, np.argmax(out_V[int(id_main[EQ, 0])-skip_*i, 512:]))
            for i in range(length):
                ax.plot(xp, np.log10(out_V[int(id_main[EQ, 0])+skip*i, :]).T, color=cm.Reds_r(i/length-0.2), lw=1.)
                if EQ == 10:
                    tips = np.append(tips, np.argmax(out_V[int(id_main[EQ, 0])+skip*i, 512:]))
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                        np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            plt.savefig('{}Figures{}Nuclei/Velocity/Nuclei_v_{}.png'.format(rel, fname, EQ), dpi=600)
            
            plt.close()
    
        print('Nuclei Velocity')
        
        
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, out_tau[int(id_main[EQ, 0])-skip_*i, :].T*(10**(-6)), color='black', lw=1.)
            for i in range(length):
                ax.plot(xp, out_tau[int(id_main[EQ, 0])+skip*i, :].T*(10**(-6)), color=cm.Reds_r(i/length-0.2), lw=1.)
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel('Shear stress (MPa)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                        np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            plt.savefig('{}Figures{}Nuclei/Tau/Nuclei_tau_{}.png'.format(rel, fname, EQ), dpi=600)
            
            plt.close()

        print('Nuclei Tau')
        
    
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, np.log10(out_state_1[int(id_main[EQ, 0])-skip_*i, :]).T, color='black', lw=1.)
            for i in range(length):
                ax.plot(xp, np.log10(out_state_1[int(id_main[EQ, 0])+skip*i, :]).T, color=cm.Reds_r(i/length-0.2), lw=1.)
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{1})|)$ (m)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                        np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            plt.savefig('{}Figures{}Nuclei/State_1/Nuclei_state1_{}.png'.format(rel, fname, EQ), dpi=600)
            
            plt.close()
        
        print('Nuclei Y1')
        
        
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, np.log10(out_state_2[int(id_main[EQ, 0])-skip_*i, :]).T, color='black', lw=1.)
            for i in range(length):
                ax.plot(xp, np.log10(out_state_2[int(id_main[EQ, 0])+skip*i, :]).T, color=cm.Reds_r(i/length-0.2), lw=1.)
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{2})|)$ (m)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                        np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            plt.savefig('{}Figures{}Nuclei/State_2/Nuclei_state2_{}.png'.format(rel, fname, EQ), dpi=600)
            
            plt.close()
        
        print('Nuclei Y2')
        
        
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, np.log10(out_state_4[int(id_main[EQ, 0])-skip_*i, :]).T, color='black', lw=1.)
            for i in range(length):
                ax.plot(xp, np.log10(out_state_4[int(id_main[EQ, 0])+skip*i, :]).T, color=cm.Reds_r(i/length-0.2), lw=1.)
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{16})|)$ (m)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                        np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            plt.savefig('{}Figures{}Nuclei/State_4/Nuclei_state4_{}.png'.format(rel, fname, EQ), dpi=600)
            
            plt.close()
        
        print('Nuclei Y4')
        
        
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, np.log10(out_state_8[int(id_main[EQ, 0])-skip_*i, :]).T, color='black', lw=1.)
            for i in range(length):
                ax.plot(xp, np.log10(out_state_8[int(id_main[EQ, 0])+skip*i, :]).T, color=cm.Reds_r(i/length-0.2), lw=1.)
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{256})|)$ (m)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                        np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            plt.savefig('{}Figures{}Nuclei/State_8/Nuclei_state8_{}.png'.format(rel, fname, EQ), dpi=600)
            
            plt.close()
        
        print('Nuclei Y8')
        
        
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, np.log10(out_state_11[int(id_main[EQ, 0])-skip_*i, :]).T, color='black', lw=1.)
            for i in range(length):
                ax.plot(xp, np.log10(out_state_11[int(id_main[EQ, 0])+skip*i, :]).T, color=cm.Reds_r(i/length-0.2), lw=1.)
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(|Y(k_{2048})|)$ (m)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                        np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))], fontsize=13)
            plt.yticks(fontsize=13)
            
            plt.tight_layout()
            plt.savefig('{}Figures{}Nuclei/State_11/Nuclei_state11_{}.png'.format(rel, fname, EQ), dpi=600)
            
            plt.close()
        
        print('Nuclei Y11')
        
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            fig, ax = plt.subplots(figsize=(5, 5))
            for i in range(length):
                ax.plot(np.array([k[0], k[3], k[15], k[255], k[-1]]),
                        np.log10(
                        2*Lf*np.array([out_state_1[int(id_main[EQ, 0])+skip*i, int(out_state_1.shape[1]//2)],
                                        out_state_2[int(id_main[EQ, 0])+skip*i, int(out_state_1.shape[1]//2)],
                                        out_state_4[int(id_main[EQ, 0])+skip*i, int(out_state_1.shape[1]//2)],
                                        out_state_8[int(id_main[EQ, 0])+skip*i, int(out_state_1.shape[1]//2)],
                                        out_state_11[int(id_main[EQ, 0])+skip*i, int(out_state_1.shape[1]//2)]
                                        ])**2
                        ),
                        color='orangered', lw=0.5, zorder=2
                        )
            for i in range(length_):
                ax.plot(np.array([k[0], k[3], k[15], k[255], k[-1]]),
                        np.log10(
                        2*Lf*np.array([out_state_1[int(id_main[EQ, 0])-skip_*i, int(out_state_1.shape[1]//2)],
                                        out_state_2[int(id_main[EQ, 0])-skip_*i, int(out_state_1.shape[1]//2)],
                                        out_state_4[int(id_main[EQ, 0])-skip_*i, int(out_state_1.shape[1]//2)],
                                        out_state_8[int(id_main[EQ, 0])-skip_*i, int(out_state_1.shape[1]//2)],
                                        out_state_11[int(id_main[EQ, 0])-skip_*i, int(out_state_1.shape[1]//2)]
                                        ])**2
                        ),
                        color='royalblue', lw=0.5, zorder=2
                        )
            ax.plot(k, np.log10(2*Lf*(Y_bar**2)), color='mediumorchid', lw=2, zorder=3, label=r'$2L_{\rm{f}}|\widebar{Y_{\rm{n}}}|^{2}$')
            ax.plot(k, np.log10(2*Lf*(beta[int(ncell/2)]*k*Y_bar/(alpha[int(ncell/2)]*Vpl + beta[int(ncell/2)]*k))**2), 
                    color='black', lw=2, zorder=4, label=r'$2L_{\rm{f}}|Y_{\rm{n}}|_{\rm{ss}}^{2}$')
            ax.set_xlabel(r'Angular wave number $k_{\rm{n}}$ (m$^{-1}$)', fontsize=13)
            ax.set_ylabel(r'Power spectral dnesity $\log_{10}(2L_{\rm{f}}|Y|^{2})$ (m$^{3}$)', fontsize=13)
            ax.set_xscale('log')
            ax.set_xlim(0.6*np.min(k), np.max(k)*1.5) 
            plt.yticks(fontsize=13)
            plt.minorticks_on()
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(zorder=1)
        
            plt.tight_layout()
            plt.savefig('{}Figures{}Nuclei/PSD_stack/PSD_stack_{}.png'.format(rel, fname, EQ), dpi=600)
            
            plt.close()
            
        print('Nuclei PSD')
    
    
        fig, ax = plt.subplots(figsize=(6.5, 5))
        V_ave = np.log10(np.mean(out_V, axis=1))
        for EQ in range(EQ_num):
            if EQ == 0:
                y = V_ave[0:id_main[0][0]]
                ttm = np.log10(out_t_sparse[id_main[0][0]] - out_t_sparse[0:id_main[0][0]])
                ax.plot(ttm, y, lw=1.5, color='black')
            else:
                y = V_ave[id_main[EQ-1][1]:id_main[EQ][0]]
                ttm = np.log10(out_t_sparse[id_main[EQ][0]] - out_t_sparse[id_main[EQ-1][1]:id_main[EQ][0]])
                ax.plot(ttm, y, lw=1.5, color='black')
        
        ax.set_xlabel(r'Time to failure $\log_{10}(t_{\rm{f}})$ (sec)', fontsize=15)
        ax.set_ylabel(r'Average slip rate $\log_{10}(\widebar{V})$ (m/s)', fontsize=15)
        ax.invert_xaxis()
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid()
        plt.tight_layout()
        plt.savefig('{}Figures{}Summary/V_ttm.png'.format(rel, fname), dpi=600)
        
        plt.close()
        
        print('Acceleration')



if EQ_before_after:
    if snap_EQ:
        plt.rcParams['xtick.direction'] = 'inout'
        plt.rcParams['ytick.direction'] = 'inout'
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['xtick.minor.size'] = 5
        plt.rcParams['ytick.minor.size'] = 5
        
        for EQ in range(EQ_num):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax_ = ax.twinx()
            ax.plot(xp, out_delta[id_main[EQ, 1], :]-out_delta[id_main[EQ, 0], :], color='black', lw=3, label='Total slip')
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel('Total slip (m)', fontsize=15)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                        np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))])
  
            ax_.plot(xp, out_tau[id_main[EQ, 0], :]*(10**(-6)), color='cornflowerblue', lw=3, label='Before event')
            ax_.plot(xp, out_tau[id_main[EQ, 1], :]*(10**(-6)), color='tomato', lw=3, label='After event')
            ax_.set_ylabel('Shear stress (MPa)', fontsize=15)
            ax_.tick_params(axis='x', labelsize=15)
            ax_.tick_params(axis='y', labelsize=15)
            plt.minorticks_on()
            plt.legend(fontsize=12)
            
            ax__ = ax.twiny()
            ax__.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*(10**(-3)), np.min(xp)/2-(hcell/4)*(10**(-3)), 0,
                        np.max(xp)/2+(hcell/4)*(10**(-3)), np.max(xp)+(hcell/2)*(10**(-3))])
            plt.minorticks_on()
            plt.tick_params(labeltop=False)
            
            plt.tight_layout()
            plt.savefig('{}Figures{}EQ_before_after/stress_ba_{}.png'.format(rel, fname, EQ), dpi=600)
            plt.close()
        
        print('Stress before and after')



if Summary:
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.direction'] = 'inout'
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['ytick.minor.size'] = 5
    
    V_ave = np.mean(out_V, axis=1)
    fig, ax = plt.subplots(figsize=(5, 5))
    for EQ in range(EQ_num):
        x = out_t_dense[id_main[EQ, 2]:id_main[EQ, 3]] - out_t_dense[id_main[EQ, 2]]
        y = np.log10(out_Pr[id_main[EQ, 2]:id_main[EQ, 3]])
        ax.plot(x, y, lw=2., color='black')
        
    ax.set_xlabel('Time since initiation to end (sec)', fontsize=15)
    ax.set_ylabel(r'Log potency rate $\log_{10}(\int_{\Sigma}{V(t){\rm{d}}\Sigma})$ (m$^2$/s)', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid()
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/Pr_comp.png'.format(rel, fname), dpi=600)
    plt.close()

    print('Potency rate')



    delta_Gc = EQcatalog[:, 6]

    fig, ax = plt.subplots(figsize=(6, 4))
    for EQ in range(EQ_num):
        ax.vlines(x=EQcatalog[EQ, 0]/yr2sec, 
                  ymin=np.min(np.log10(delta_Gc)), ymax=np.log10(delta_Gc[EQ]), color='dimgray', lw=1, zorder=1)
        ax.scatter(EQcatalog[EQ, 0]/yr2sec, np.log10(delta_Gc[EQ]), marker='*', s=200, facecolor='black',
                   edgecolor='gray', linewidth=0.5, zorder=2)
    ax.set_xlabel('Simulation time (year)', fontsize=15)
    ax.set_ylabel(r'Average coseismic slip $\log_{10}(\widebar{s})$ (m)', fontsize=12)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/MT.png'.format(rel, fname), dpi=600)
    
    plt.close()
    
    print('MT')


    fig, ax = plt.subplots(figsize=(5, 5))
    x_ll = np.min(np.log10(delta_Gc))
    x_ul = np.max(np.log10(delta_Gc))
    ax.hist(np.log10(delta_Gc), range=(x_ll, x_ul), 
            bins=30, facecolor='black', edgecolor='silver', lw=1.5, zorder=2)
    ax.set_xlabel(r'Average coseismic slip $\log_{10}(\widebar{s})$', fontsize=15)
    ax.set_ylabel('Event count', fontsize=15)
    ax.set_yscale('log')
    ax.set_xlim(x_ll-0.01, x_ul+0.01)
    ax.set_ylim(10**(-1), 10**(2))
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/GR.png'.format(rel, fname), dpi=600)
    
    plt.close()

    print('GR')

    fig, ax = plt.subplots(figsize=(5, 5))
    duration = EQcatalog[:, 1] - EQcatalog[:, 0]
    x = np.log10(EQcatalog[:, 3])
    y = np.log10(duration)
    ax.scatter(x, y, facecolor='None', edgecolor='black', lw=2, s=100, zorder=2)
    ax.scatter(np.log10(err_source[:, 1]), y, facecolor='None', edgecolor='deeppink')
    ax.scatter(np.log10(err_source[:, 5]), y, facecolor='None', edgecolor='slateblue')
    ax.set_xlabel(r'Log potency $\log_{10}(\int_{\Sigma}\int_{T}V {\rm{d}}t {\rm{d}}\Sigma)$ (m$^2$)', fontsize=15)
    ax.set_ylabel(r'Log event duration $\log_{10}(T)$ (sec)', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/D_P.png'.format(rel, fname), dpi=600)

    plt.close()

    print('Potency-duration')
    
    fig, ax = plt.subplots(figsize=(5, 5))
    duration = EQcatalog[:, 1] - EQcatalog[:, 0]
    x = np.log10(delta_Gc)
    y = EQcatalog[:, 4]*(10**(-6))
    ax.scatter(x, y, facecolor='None', edgecolor='orangered', lw=2, s=100, zorder=2)
    ax.scatter(np.log10(err_source[:, 3]), err_source[:, 0]*(10**(-6)), facecolor='None', edgecolor='deeppink', lw=2, s=100, zorder=2)
    ax.scatter(np.log10(err_source[:, 7]), err_source[:, 4]*(10**(-6)), facecolor='None', edgecolor='slateblue', lw=2, s=100, zorder=2)
    ax.set_xlabel(r'Average coseismic slip $\log_{10}(\widebar{s})$ (m)', fontsize=15)
    ax.set_ylabel(r'Stress drop $\Delta\sigma$ (MPa)', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/Stress_drop.png'.format(rel, fname), dpi=600)

    plt.close()

    print('Slip-stress drop')
    
    fig, ax = plt.subplots(figsize=(5, 5))
    duration = EQcatalog[:, 1] - EQcatalog[:, 0]
    x = np.log10(delta_Gc)
    y = np.log10(EQcatalog[:, 9])
    ax.scatter(x, y, facecolor='None', edgecolor='lightseagreen', lw=2, s=100, zorder=2)
    ax.set_xlabel(r'Average coseismic slip $\log_{10}(\widebar{s})$ (m)', fontsize=15)
    ax.set_ylabel(r'Log Radiation energy $E_{\rm{R}}$ (J)', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/ER.png'.format(rel, fname), dpi=600)
    #plt.savefig('{}Figures{}Summary/D_M0.eps'.format(rel, fname))

    plt.close()

    print('Radiation energy')
    
    
    fig, ax = plt.subplots(figsize=(5, 5))
    duration = EQcatalog[:, 1] - EQcatalog[:, 0]
    x = np.log10(delta_Gc)
    y = np.log10(EQcatalog[:, 8])
    ax.scatter(x[:], y[:], facecolor='None', edgecolor='limegreen', lw=2, s=100, zorder=2)    
    ax.set_xlabel(r'Average coseismic slip $\log_{10}(\widebar{s})$ (m)', fontsize=15)
    ax.set_ylabel(r'Log Available energy $E_{\rm{A}}$ (J)', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/EA.png'.format(rel, fname), dpi=600)
    #plt.savefig('{}Figures{}Summary/D_M0.eps'.format(rel, fname))

    plt.close()

    print('Available energy')
    
    
    fig, ax = plt.subplots(figsize=(5, 5))
    duration = EQcatalog[:, 1] - EQcatalog[:, 0]
    x = np.log10(delta_Gc)
    y = np.log10(EQcatalog[:, 9]/EQcatalog[:, 8])
    ax.scatter(x, y, facecolor='None', edgecolor='darkviolet', lw=2, s=100, zorder=2)
    ax.set_xlabel(r'Average coseismic slip $\log_{10}(\widebar{s})$ (m)', fontsize=15)
    ax.set_ylabel(r'Log radiation efficiency $\eta_{\rm{R}}$', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/etaR.png'.format(rel, fname), dpi=600)
    #plt.savefig('{}Figures{}Summary/D_M0.eps'.format(rel, fname))

    plt.close()

    print('Radiation efficiency')

    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.log10(delta_Gc)
    y = np.log10(EQcatalog[:, 5])
    ax.scatter(x, y, facecolor='None', lw=2, s=100, zorder=2, edgecolor='deeppink')
    ax.scatter(np.log10(err_source[:, 3]), np.log10(err_source[:, 2]), facecolor='None', lw=2, s=100, zorder=2, edgecolor='black')
    ax.scatter(np.log10(err_source[:, 7]), np.log10(err_source[:, 6]), facecolor='None', lw=2, s=100, zorder=2, edgecolor='orangered')
    ax.set_xlabel(r'Log coseismic slip $\log_{10}(\widebar{s})$ (m)', fontsize=15)
    ax.set_ylabel(r'Log average fracture energy $G_{\rm{c}}$ (J/m$^2$)', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/Gc_S.png'.format(rel, fname), dpi=600)
    #plt.savefig('{}Figures{}Summary/Gc_S.eps'.format(rel, fname))

    plt.close()

    print('Fracture energy')