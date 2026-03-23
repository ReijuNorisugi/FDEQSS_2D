# -*- coding: utf-8 -*-
"""
Code for analysing dynamic earthquake sequence simulation.
This is valid for simulation with standard RSF law.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260320.
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

fname = '/reg_RSF_AG/test/'   # Please type directory name.

current = os.getcwd()
# Set the path to the host directory.
target = "/Users/rn/Documents/FDEQSS_2D"
rel = ""
while current != target and current != os.path.dirname(current):
    current = os.path.dirname(current)
    rel = "../" + rel

# Choose which to plot. When EQ_num is large, Dyn_snap is time-consuming.
Slip_V_cmap     = True
Check           = False
Trajectory      = False
Dyn_snap        = False
EQ_before_after = False
Summary         = False

yr2sec = 365.*24.*60.*60.
day2sec = 24.*60.*60.
min2sec = 60.

# Duration for coseismic snapshots.
duration = 0.5 # sec.
# Duration for aseismic snapshots.
duration_ = 10.  # year.
duration_ = duration_ * yr2sec

# How many events you want to show in a snapshot?
plot_num = 9
reduce = 1  # You can reduce data points for contour if it is too heavy.

###########   Load parameters used in simulation.   ###########
Conditions = utils.load('{}Output{}Conditions.pkl'.format(rel, fname))
Medium = utils.load('{}Output{}Medium.pkl'.format(rel, fname))
FaultParams = utils.load('{}Output{}FaultParams.pkl'.format(rel, fname))

# Generate directories to save output figures.
os.makedirs('{}Figures{}Slip_V_cmap'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Slip_V_contour'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/Tau_V'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/Tau_slip'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Trajectory/V_state'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Dyn_snap'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Dyn_snap/SlipRate'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Dyn_snap/Tau'.format(rel, fname), exist_ok=True)
os.makedirs('{}Figures{}Dyn_snap/State'.format(rel, fname), exist_ok=True)
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
downsample = Conditions['downsample']

Cs = Medium['Cs']
Cp = Medium['Cp']
mu = Medium['mu']
nu = Medium['nu']
eta = Medium['eta']
f0 = Medium['f0']
V0 = Medium['V0']
Vpl = Medium['Vpl']
H = Medium['H']
Z = Medium['Z']
tau_rate = Medium['tau_rate']

beta_min = Medium['beta_min']
lam = Medium['lam']
Nele = Medium['Nele']
hcell = Medium['hcell']
nperi = Medium['nperi']
kn = Medium['kn']
xp = Medium['xp'] * 1.e-3 # km
xp = xp[::downsample]
ncell = Medium['ncell']
dtmin = Medium['dtmin']
Tw = Medium['Tw']

a = FaultParams['a']
b = FaultParams['b']
L = FaultParams['L']
sigma = FaultParams['sigma']
xi = FaultParams['xi']
v_cos = np.min(2. * Cs * a * sigma / mu)
##############################################################

##########   Load counting for file opening   #############
id_count = np.load('{}Output{}id_count.npy'.format(rel, fname))
id_count_st = np.load('{}Output{}id_count_st.npy'.format(rel, fname))
EQ_num = np.load('{}Output{}EQ_num.npy'.format(rel, fname))
id_station = np.load('{}Output{}id_station.npy'.format(rel, fname))
t_prep = np.load('{}Output{}t_prep.npy'.format(rel, fname))
t_wh = np.load('{}Output{}t_wh.npy'.format(rel, fname))
########################################################

###############################     Unpack binary outputs    ##################################
out_t_sparse = np.fromfile('{}Output{}out_t_sparse.bin'.format(rel, fname), dtype=np.float64)
out_t_dense  = np.fromfile('{}Output{}out_t_dense.bin'.format(rel, fname), dtype=np.float64)

out_delta = np.fromfile('{}Output{}out_delta.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_V     = np.fromfile('{}Output{}out_V.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_tau   = np.fromfile('{}Output{}out_tau.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_state = np.fromfile('{}Output{}out_state.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))
out_Gcd   = np.fromfile('{}Output{}out_Gcd.bin'.format(rel, fname), dtype=np.float64).reshape(-1, int(ncell/downsample))

out_Pr = np.fromfile('{}Output{}out_Pr.bin'.format(rel, fname), dtype=np.float64)

EQcatalog = np.fromfile('{}Output{}out_catalog.bin'.format(rel, fname), dtype=np.float64).reshape(-1, 11)
id_main   = np.fromfile('{}Output{}out_idmain.bin'.format(rel, fname), dtype=np.int64).reshape(-1, 4)

if outerror:
    out_error = np.fromfile('{}Output{}out_error.bin'.format(rel, fname), dtype=np.float64)
    out_dt    = np.fromfile('{}Output{}out_dt.bin'.format(rel, fname), dtype=np.float64)

if dense:
    out_delta_st = np.fromfile('{}Output{}out_delta_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
    out_V_st     = np.fromfile('{}Output{}out_V_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
    out_tau_st   = np.fromfile('{}Output{}out_tau_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
    out_state_st = np.fromfile('{}Output{}out_state_station.bin'.format(rel, fname), dtype=np.float64).reshape(-1, len(id_station))
#########################################################################################################################

print('')
print('Preparation time {} sec'.format(t_prep))
if t_wh/day2sec >= 0.1:
    print('Consuming time {} day'.format(t_wh/day2sec))
elif t_wh/day2sec < 0.1 and t_wh/min2sec >= 5.:
    print('Consuming time {} min'.format(t_wh/min2sec))
else:
    print('Consuming time {} sec'.format(t_wh))
print('{} time steps'.format(id_count_st))
print('{} quakes'.format(EQ_num))
print('')

log_V = np.log10(out_V/Vpl)
thres = np.log10(v_cos/Vpl)
# vmin and vmax determines velocity range in color map.
vmin = 0.
vmax = np.max(log_V)
norm_V = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

if EQ_num != 0:
    colors = [(0, '#b3b3b3'), 
              (norm_V(thres)*1/5, '#47585c'), 
              (norm_V(thres)*2/5, '#3d4b99'), 
              (norm_V(thres)*3/5, '#00a1e9'), 
              (norm_V(thres)*4/5, '#e8e8e8'), 
              (norm_V(thres), '#ffccbb'), 
              ((norm_V(thres)+1)*0.5, '#ff7f40'), 
              (1, '#c70000')]
else:
    colors = [(0, '#b3b3b3'), 
              (0.2, '#47585c'), 
              (0.4, '#3d4b99'), 
              (0.6, '#00a1e9'), 
              (0.8, '#e8e8e8'), 
              (1, '#ff7f40')]

blue_orange = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

# Color map for velocity.
norm_V = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
fig, cbar = plt.subplots(figsize=(5, 0.3))
mpl.colorbar.Colorbar(
ax=cbar,
mappable=cm.ScalarMappable(norm=norm_V, cmap=blue_orange),
orientation='horizontal',
).set_label(r'$\log_{10}(V/V_{\rm{pl}})$', fontsize=15)
plt.xticks(fontsize=13)
plt.savefig('{}Figures{}Slip_V_cmap/colormap_v_.png'.format(rel, fname), dpi=600, bbox_inches='tight')
plt.close()



if Slip_V_cmap:
    if EQ_num == 0:
        plot_num = np.max(out_delta)                    # SLip included in a single figure.
        plot_range = int(np.max(out_delta)//plot_num)   # Number of figures.
    else:
        plot_range = round(EQ_num//plot_num)            # Number of figures.

    # Plot color map of velocity with slip snapshots and Lapusta plot.
    for j in range(plot_range):
        if not sparse:
            break
        else:
            max_delta = np.max(out_delta, axis=1)
            min_delta = np.min(out_delta, axis=1)

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
                plt.tricontourf(triangulation, color, cmap=blue_orange, levels=200, vmin=vmin, vmax=vmax)
            elif mode == 'IV':
                figsize = (26, 11)
            
                points = np.vstack([xp_array.ravel(), out_delta[begin:end, ::reduce].T.ravel()]).T
                tri = Delaunay(points)
                triangulation = Triangulation(points[:, 0], points[:, 1], tri.simplices)
            
                fig, ax = plt.subplots(figsize=figsize)
                plt.tricontourf(triangulation, color, cmap=blue_orange, levels=200, vmin=vmin, vmax=vmax)
            else:
                figsize = (9, 18)
            
                points = np.vstack([xp_array.ravel(), out_delta[begin:end, ::reduce].T.ravel()]).T
                tri = Delaunay(points)
                triangulation = Triangulation(points[:, 0], points[:, 1], tri.simplices)
            
                fig, ax = plt.subplots(figsize=figsize)
                plt.tricontourf(triangulation, color, cmap=blue_orange, levels=200, vmin=vmin, vmax=vmax)
        
            if EQ_num != 0:
                id_cos = np.empty(0, dtype=np.int64)
                for l in range(plot_num):
                    for i in range(1000):
                        if out_t_sparse[id_begin[l]] + duration*i >= out_t_sparse[id_end[l]]:
                            break
                        else:
                            id_cos = np.append( id_cos, np.argmin( np.abs(out_t_sparse - out_t_sparse[id_begin[l]]-duration*i) ) )
        
            id_ass = np.empty(0, dtype=np.int64)
            id_ass = np.append(id_ass, 0)
            for i in range(1000):
                if out_t_sparse[begin]+duration_*i <= out_t_sparse[end]:
                    id_ass = np.append(id_ass, np.argmin( np.abs(out_t_sparse - out_t_sparse[begin] - duration_*i)))
        
        
            if mirror:
                if EQ_num != 0:
                    ax.plot(out_delta[id_cos].T, xp, color='black', linestyle='dashed', lw=4, label='Every {}s'.format(duration))
                    ax.plot(out_delta[id_end].T, xp, color='black', linestyle='solid', lw=6, label='Total slip')
                    ax.plot(out_delta[id_ass].T, xp, color='black', linestyle='solid', lw=4, label='Every {}yr'.format(int(duration_/yr2sec)))
                else:
                    ax.plot(out_delta[id_ass].T, xp, color='black', linestyle='solid', lw=4, label='Every {}yr'.format(int(duration_/yr2sec)))
                ax.set_xlabel('Cumulative slip (m)', fontsize=35)
                ax.set_ylabel('Distance along fault (km)', fontsize=35)
                ax.set_xlim(np.min(out_delta[begin:end]), np.max(out_delta[begin:end]))
                ax.set_ylim(np.min(xp), np.max(xp))
                ax.invert_yaxis()
                plt.xticks(fontsize=30)
                plt.yticks([0, np.max(xp)/2+(hcell/4)*1.e-3, np.max(xp)+(hcell/2)*1.e-3], fontsize=30)
            else:
                if EQ_num != 0:
                    ax.plot(xp, out_delta[id_cos].T, color='black', linestyle='dashed', lw=4, label='Every {}s'.format(duration))
                    ax.plot(xp, out_delta[id_end].T, color='black', linestyle='solid', lw=6, label='Total slip')
                    ax.plot(xp, out_delta[id_ass].T, color='black', linestyle='solid', lw=4, label='Every {}yr'.format(int(duration_/yr2sec)))
                else:
                    ax.plot(xp, out_delta[id_ass].T, color='black', linestyle='solid', lw=4, label='Every {}yr'.format(int(duration_/yr2sec)))
                ax.set_xlabel('Distance along fault (km)', fontsize=35)
                ax.set_ylabel('Cumulative slip (m)', fontsize=35)
                ax.set_xlim(np.min(xp), np.max(xp))
                ax.set_ylim(np.min(out_delta[begin:end]), np.max(out_delta[begin:end]))
                plt.xticks([np.min(xp)-(hcell/2)*1.e-3, np.min(xp)/2-(hcell/4)*1.e-3, 0,
                            np.max(xp)/2+(hcell/4)*1.e-3, np.max(xp)+(hcell/2)*1.e-3], fontsize=30)
                plt.yticks(fontsize=30)
            handles, labels = plt.gca().get_legend_handles_labels()
            if EQ_num == 0:
                    plt.legend([handles[0]], [labels[0]],
                               loc='lower right', fontsize=30, framealpha=1)
            else:
                    plt.legend([handles[0], handles[-1]], [labels[0], labels[-1]],
                               loc='lower right', fontsize=30, framealpha=1)
            plt.tight_layout()
            plt.savefig('{}Figures{}Slip_V_cmap/Slip_{}.png'.format(rel, fname, j), dpi=600)
            plt.close()

            
            figsize = (3.25, 1.375)
            fig, ax = plt.subplots(figsize=figsize)
            plt.tricontourf(triangulation, color, cmap=blue_orange, levels=64, vmin=vmin, vmax=vmax)
            ax.plot(out_delta[id_cos].T, xp, color='black', linestyle='dashed', lw=0.5, label='Every {}s'.format(duration))
            ax.plot(out_delta[id_end].T, xp, color='black', linestyle='solid', lw=0.5, label='Total slip')
            ax.plot(out_delta[id_ass].T, xp, color='black', linestyle='solid', lw=0.5, label='Every {}yr'.format(int(duration_/yr2sec)))
            ax.set_xlim(10, 30)
            ax.set_ylim(np.min(xp), np.max(xp))
            ax.invert_yaxis()
            plt.xticks([])
            plt.yticks([])
            ax.axis('off')
            fig.subplots_adjust(0, 0, 1, 1)
            plt.savefig(
                '{}Figures{}Slip_V_cmap/Slip_tmp.png'.format(rel, fname),
                dpi=450,
                bbox_inches='tight',
                pad_inches=0,
            )
            plt.close()




            #############################################
        
            fig, ax = plt.subplots(figsize=figsize)
            if mirror:
                if EQ_num != 0:
                    ax.plot(out_delta[id_cos].T, xp, color='tomato', linestyle='solid', lw=4, label='Every {}s'.format(duration))
                    ax.plot(out_delta[id_end].T, xp, color='black', linestyle='solid', lw=6, label='Total slip')
                    ax.plot(out_delta[id_ass].T, xp, color='royalblue', linestyle='solid', lw=4, label='Every {}yr'.format(int(duration_/yr2sec)))
                else:
                    ax.plot(out_delta[id_ass].T, xp, color='blue', linestyle='solid', lw=4, label='Every {}yr'.format(int(duration_/yr2sec)))
                ax.set_xlim(np.min(out_delta[begin:end]), np.max(out_delta[begin:end]))
                ax.set_ylim(np.min(xp), np.max(xp))
                ax.set_xlabel('Cumulative slip (m)', fontsize=25)
                ax.set_ylabel('Distance along strike (km)', fontsize=25)
                ax.invert_yaxis()
                plt.xticks(fontsize=30)
                plt.yticks([0, np.max(xp)/2+(hcell/4)*1.e-3, np.max(xp)+(hcell/2)*1.e-3], fontsize=30)
            else:
                if EQ_num != 0:
                    ax.plot(xp, out_delta[id_cos].T, color='tomato', linestyle='solid', lw=4, label='Every {}s'.format(duration))
                    ax.plot(xp, out_delta[id_end].T, color='black', linestyle='solid', lw=6, label='Total slip')
                    ax.plot(xp, out_delta[id_ass].T, color='royalblue', linestyle='solid', lw=4, label='Every {}yr'.format(int(duration_/yr2sec)))
                else:
                    ax.plot(xp, out_delta[id_ass].T, color='blue', linestyle='solid', lw=4, label='Every {}yr'.format(int(duration_/yr2sec)))
                ax.set_xlim(np.min(xp), np.max(xp))
                ax.set_ylim(np.min(out_delta[begin:end]), np.max(out_delta[begin:end]))
                ax.set_xlabel('Distance along strike (km)', fontsize=25)
                ax.set_ylabel('Cumulative slip (m)', fontsize=25)
                plt.xticks([np.min(xp)-(hcell/2)*1.e-3, np.min(xp)/2-(hcell/4)*1.e-3, 0,
                            np.max(xp)/2+(hcell/4)*1.e-3, np.max(xp)+(hcell/2)*1.e-3], fontsize=30)
                plt.yticks(fontsize=30)
            handles, labels = plt.gca().get_legend_handles_labels()
            if EQ_num == 0:
                plt.legend([handles[0]], [labels[0]],
                            loc='lower right', fontsize=30, framealpha=1)
            else:
                plt.legend([handles[0], handles[-1]], [labels[0], labels[-1]],
                            loc='lower right', fontsize=30, framealpha=1)
            plt.tight_layout()
            plt.savefig('{}Figures{}Slip_V_contour/Slip_{}.png'.format(rel, fname, j), dpi=600)
            plt.close()
        

if Check:
    if outerror:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(out_t_dense/yr2sec, out_error, color='black', lw=3)
        ax.set_xlabel('Simulation time (year)', fontsize=15)
        ax.set_ylabel(r'Error $\epsilon$', fontsize=15)
        ax.set_yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig('{}Figures{}Check/Error.png'.format(rel, fname), dpi=600)
        
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(out_t_dense/yr2sec, out_dt, color='black', lw=3)
        ax.set_xlabel('Simulation time (year)', fontsize=15)
        ax.set_ylabel(r'Time step $dt$ (sec)', fontsize=15)
        ax.set_yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig('{}Figures{}Check/Time_steps.png'.format(rel, fname), dpi=600)
        
        plt.close()
  
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(out_t_dense/yr2sec, out_error/out_dt, color='black', lw=3)
        ax.set_xlabel('Simulation time (year)', fontsize=15)
        ax.set_ylabel(r'Ratio $\epsilon/dt$ (/sec)', fontsize=15)
        ax.set_yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig('{}Figures{}Check/Ratio.png'.format(rel, fname), dpi=600)
        
        plt.close()


if Trajectory:
    if dense:
        V_plot = 10**(np.linspace(-16, 2, 100))
        tau_ss = np.zeros((len(id_station), len(V_plot)), dtype=np.float64)
        for i in range(len(id_station)):
            tau_ss[i, :] = f0*sigma[id_station[i]] + (a[id_station[i]]-b[id_station[i]]) * sigma[id_station[i]] * np.log(V_plot/V0)
        
        y_ul = round(np.max(out_tau_st*1.e-6))
        y_ll = round(np.min(out_tau_st*1.e-6))
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(np.log10(V_plot), tau_ss[stat, :]*1.e-6,
                    color='darkorange', lw=3, label=r'$\tau_{\rm{ss}}$')
            ax.plot(np.log10(out_V_st[:, stat]), out_tau_st[:, stat]*1.e-6,
                    color='black', lw=3, label='Trajectory')
            ax.set_xlim(-15, 2)
            ax.set_ylim(y_ll-1, y_ul+1)
            ax.set_xlabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=15)
            ax.set_ylabel('Shear stress (MPa)', fontsize=15)
            
            plt.xticks([-14, -12, -10, -8, -6, -4, -2, 0, 2], fontsize=13)
            plt.yticks(np.arange(y_ll, y_ul, 4), fontsize=13)
            plt.minorticks_on()
            plt.legend(loc='upper left', fontsize=13, framealpha=1)
            plt.grid()
            plt.tick_params(which='both',
                            top=True, labeltop=False, 
                            right=True, labelright=False)
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/Tau_V/TV_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
  
        print('Trajectory TV')
        
        state_ss = np.zeros((len(id_station), len(V_plot)), dtype=np.float64)
        for i in range(len(id_station)):
            state_ss[i, :] = L[id_station[i]] / V_plot
        
        y_ul = round(np.max(np.log10(out_state_st)))
        y_ll = round(np.min(np.log10(out_state_st)))
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(np.log10(V_plot), np.log10(state_ss[stat, :]),
                    color='darkorange', lw=3, label=r'$\theta_{\rm{ss}}$')
            ax.plot(np.log10(out_V_st[:, stat]), np.log10(out_state_st[:, stat]),
                    color='black', lw=3, label='Trajectory')
            ax.set_xlim(-15, 2)
            ax.set_ylim(y_ll-1, y_ul+1)
            ax.set_xlabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(\theta)$ (sec)', fontsize=15)
            
            plt.xticks([-14, -12, -10, -8, -6, -4, -2, 0, 2], fontsize=13)
            plt.yticks(np.arange(y_ll, y_ul+2, 2), fontsize=13)
            plt.minorticks_on()
            plt.legend(loc='lower left', fontsize=13, framealpha=1)
            plt.grid()
            plt.tick_params(which='both',
                            top=True, labeltop=False, 
                            right=True, labelright=False)
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/V_state/VT_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
            
        print('Trajectory VT')
        
        
        for stat in range(len(id_station)):
            fig, ax = plt.subplots(figsize=(5, 5))
            x_ul = np.empty(0)
            x_ll = np.empty(0)
            y_ul = np.empty(0)
            y_ll = np.empty(0)
            for EQ in range(EQ_num):
                x = out_delta_st[id_main[EQ, 2]:id_main[EQ, 3], stat] - out_delta_st[id_main[EQ, 2], stat]
                y = out_tau_st[id_main[EQ, 2]:id_main[EQ, 3], stat]*1.e-6
                ax.plot(x, y, color='black', lw=3)
                
                x_ul = np.append(x_ul, round(10*np.max(x)))
                x_ll = np.append(x_ll, round(10*np.min(x)))
                y_ul = np.append(y_ul, round(np.max(y)))
                y_ll = np.append(y_ll, round(np.min(y)))
            x_ul = np.max(x_ul)/10
            x_ll = np.min(x_ll)/10
            y_ul = np.max(y_ul)
            y_ll = np.min(y_ll)
            
            ax.set_xlabel('Coseismic slip (m)', fontsize=15)
            ax.set_ylabel('Shear stress (MPa)', fontsize=15)
            if x_ul <= 0.5:
                interval = 0.05
            else:
                interval = 0.1
            if x_ul <= 0.1:
                interval = 0.02
            ax.set_xlim(x_ll-interval/2, x_ul+interval)
            ax.set_ylim(y_ll-1., y_ul+1)
            
            plt.xticks(fontsize=13)
            plt.yticks(np.arange(y_ll, y_ul+2, 2), fontsize=13)
            plt.grid()
            plt.tick_params(which='both',
                            top=True, labeltop=False, 
                            right=True, labelright=False)
            plt.tight_layout()
            plt.savefig('{}Figures{}Trajectory/Tau_Slip/TS_{}.png'.format(rel, fname, stat), dpi=600)
            
            plt.close()
            
        print('Trajctory TS')



if EQ_num == 0:
    sys.exit()
else:
    pass

if Dyn_snap:
    if sparse:
        skip = 2
        skip_ = 2
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            y_ul = round(np.max(np.log10(out_V)))
            y_ll = round(np.log10(Vpl))
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, np.log10(out_V[int(id_main[EQ, 0])-skip_*i, :]).T, color='black', lw=1.)
            for i in range(length):
                ax.plot(xp, np.log10(out_V[int(id_main[EQ, 0])+skip*i, :]).T, color=cm.Reds_r(i/length-0.2), lw=1.)
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel(r'Slip rate $\log_{10}(V)$ (m/s)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            ax.set_ylim(y_ll-1, y_ul+1)
            plt.xticks([np.min(xp)-(hcell/2)*1.e-3, np.min(xp)/2-(hcell/4)*1.e-3, 0,
                        np.max(xp)/2+(hcell/4)*1.e-3, np.max(xp)+(hcell/2)*1.e-3], fontsize=13)
            plt.yticks(np.arange(y_ll, y_ul+1, 2), fontsize=13)
            plt.tick_params(which='both',
                            top=True, labeltop=False, 
                            right=True, labelright=False)
            plt.tight_layout()
            plt.savefig('{}Figures{}Dyn_snap/SlipRate/Dyn_snap_v_{}.png'.format(rel, fname, EQ), dpi=600)
            plt.close()

        print('Dyn_snap Velocity')

        
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            y_ul = round(np.max(np.log10(out_state)))
            y_ll = round(np.min(np.log10(out_state)))
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, np.log10(out_state[int(id_main[EQ, 0])-skip_*i, :]).T, color='black', lw=1.)
            for i in range(length):
                ax.plot(xp, np.log10(out_state[int(id_main[EQ, 0])+skip*i, :]).T, color=cm.Reds_r(i/length-0.2), lw=1.)
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel(r'State variable $\log_{10}(\theta)$ (s)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            ax.set_ylim(y_ll-1, y_ul+1)
            plt.xticks([np.min(xp)-(hcell/2)*1.e-3, np.min(xp)/2-(hcell/4)*1.e-3, 0,
                        np.max(xp)/2+(hcell/4)*1.e-3, np.max(xp)+(hcell/2)*1.e-3], fontsize=13)
            plt.yticks(np.arange(y_ll, y_ul+2, 2), fontsize=13)
            plt.tick_params(which='both',
                            top=True, labeltop=False, 
                            right=True, labelright=False)
            plt.tight_layout()
            plt.savefig('{}Figures{}Dyn_snap/State/Dyn_snap_state_{}.png'.format(rel, fname, EQ), dpi=600)
            plt.close()
  
        print('Dyn_snap State')
        
        
        for EQ in range(EQ_num):
            length = id_main[EQ, 1] - id_main[EQ, 0]
            length = round(length/skip)
            for i in range(1000):
                if np.max(out_V[int(id_main[EQ, 0])-i, :]) < 10*Vpl:
                    length_ = i
                    break
            length_ = round(length_/skip_)
            
            y_ul = round(np.max(out_tau*1.e-6))
            y_ll = round(np.min(out_tau*1.e-6))
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(length_):
                ax.plot(xp, out_tau[int(id_main[EQ, 0])-skip_*i, :].T*1.e-6, color='black', lw=1.)
            for i in range(length):
                ax.plot(xp, out_tau[int(id_main[EQ, 0])+skip*i, :].T*1.e-6, color=cm.Reds_r(i/length-0.2), lw=1.)
            ax.set_xlabel('Distance along fault (m)', fontsize=15)
            ax.set_ylabel('Shear stress (MPa)', fontsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            ax.set_ylim(y_ll-2, y_ul+2)
            plt.xticks([np.min(xp)-(hcell/2)*1.e-3, np.min(xp)/2-(hcell/4)*1.e-3, 0,
                        np.max(xp)/2+(hcell/4)*1.e-3, np.max(xp)+(hcell/2)*1.e-3], fontsize=13)
            plt.yticks(np.arange(y_ll, y_ul+2, 2), fontsize=13)
            plt.tick_params(which='both',
                            top=True, labeltop=False, 
                            right=True, labelright=False)
            plt.tight_layout()
            plt.savefig('{}Figures{}Dyn_snap/Tau/Dyn_snap_tau_{}.png'.format(rel, fname, EQ), dpi=600)
            plt.close()

        print('Dyn_snap Tau')
    
        fig, ax = plt.subplots(figsize=(5, 5))
        V_ave = np.log10(np.mean(out_V, axis=1))
        y_ll = round(np.min(V_ave))
        
        y_ul = np.empty(0)
        for EQ in range(EQ_num):
            if EQ == 0:
                y = V_ave[0:id_main[0][0]]
                ttm = np.log10(out_t_sparse[id_main[0][0]] - out_t_sparse[0:id_main[0][0]])
                ax.plot(ttm, y, lw=1.5, color='black')
            else:
                y = V_ave[id_main[EQ-1][1]:id_main[EQ][0]]
                ttm = np.log10(out_t_sparse[id_main[EQ][0]] - out_t_sparse[id_main[EQ-1][1]:id_main[EQ][0]])
                ax.plot(ttm, y, lw=1.5, color='black')
                
            y_ul = np.append(y_ul, round(np.max(y))+1)
        y_ul = np.max(y_ul)
    
        ax.set_xlabel(r'Time to failure $\log_{10}(t_{\rm{f}})$ (sec)', fontsize=15)
        ax.set_ylabel(r'Average slip rate $\log_{10}(\widebar{V})$ (m/s)', fontsize=15)
        ax.set_xlim(-4, 10)
        ax.set_ylim(y_ll-0.5, y_ul+0.5)
        ax.invert_xaxis()
        plt.xticks(np.arange(-4, 12, 2), fontsize=13)
        plt.yticks(np.arange(y_ll, y_ul+0.5, 2), fontsize=13)
        plt.grid()
        plt.tick_params(which='both',
                            top=True, labeltop=False, 
                            right=True, labelright=False)
        plt.tight_layout()
        plt.savefig('{}Figures{}Summary/V_TTF.png'.format(rel, fname), dpi=600)
        plt.close()
    
        print('Acceleration')

        

if EQ_before_after:
    if snap_EQ:
        for EQ in range(EQ_num):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax_ = ax.twinx()
            ax.plot(xp, out_delta[id_main[EQ, 1], :]-out_delta[id_main[EQ, 0], :], color='black', lw=3, label='Total slip')
            ax.set_xlabel('Distance along fault (km)', fontsize=15)
            ax.set_ylabel('Total slip (m)', fontsize=15)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_xlim(np.min(xp), np.max(xp))
            plt.xticks([np.min(xp)-(hcell/2)*1.e-3, np.min(xp)/2-(hcell/4)*1.e-3, 0,
                        np.max(xp)/2+(hcell/4)*1.e-3, np.max(xp)+(hcell/2)*1.e-3])
            
            ax_.plot(xp, out_tau[id_main[EQ, 0], :]*1.e-6, color='cornflowerblue', lw=3, label='Before event')
            ax_.plot(xp, out_tau[id_main[EQ, 1], :]*1.e-6, color='tomato', lw=3, label='After event')
            ax_.set_ylabel('Shear stress (MPa)', fontsize=15)
            ax_.tick_params(axis='x', labelsize=15)
            ax_.tick_params(axis='y', labelsize=15)
            plt.minorticks_on()
            plt.legend(fontsize=12)
            
            plt.tick_params(which='both',
                            top=True, labeltop=False)
            
            plt.tight_layout()
            plt.savefig('{}Figures{}EQ_before_after/stress_ba_{}.png'.format(rel, fname, EQ), dpi=600)
            plt.close()
        
        print('Snap EQ before after')



if Summary:
    EQ_start = EQcatalog[:, 0] # Start time of EQ.
    EQ_end   = EQcatalog[:, 1] # End time of EQ.
    Hypo     = EQcatalog[:, 2] # Hypocenter.
    Pot      = EQcatalog[:, 3] # Seismic potency.
    sigma_M  = EQcatalog[:, 4] # Moment-based stress drop.
    sigma_E  = EQcatalog[:, 10] # Energy-based stress drop.
    Gc       = EQcatalog[:, 5] # Fracture energy.
    slip_ave = EQcatalog[:, 6] # Average coseismic slip.
    EG       = EQcatalog[:, 7] # Energy dissipation.
    Ea       = EQcatalog[:, 8] # Available energy.
    ER       = EQcatalog[:, 9] # Radiated energy.
    
    x_ul = np.empty(0)
    x_ll = np.empty(0)
    y_ul = np.empty(0)
    y_ll = np.empty(0)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    for EQ in range(EQ_num):
        x = out_t_dense[id_main[EQ, 2]:id_main[EQ, 3]] - out_t_dense[id_main[EQ, 2]]
        y = np.log10(out_Pr[id_main[EQ, 2]:id_main[EQ, 3]])
        ax.plot(x, y, lw=2., color='black')
        
        x_ul = np.append(x_ul, round(np.max(x)))
        x_ll = np.append(x_ll, round(np.min(x)))
        y_ul = np.append(y_ul, round(np.max(y)))
        y_ll = np.append(y_ll, round(np.min(y)))
    x_ul = np.max(x_ul)
    x_ll = np.min(x_ll)
    y_ul = np.max(y_ul)
    y_ll = np.min(y_ll)
    
    ax.set_xlabel('Time since initiation to end (sec)', fontsize=15)
    ax.set_ylabel(r'Log potency rate (m$^2$/s)', fontsize=15)
    ax.set_xlim(-0.1, x_ul+1)
    ax.set_ylim(y_ll-0.2, y_ul+0.2)
    if x_ul >= 10:
        interval = 5
    elif x_ul <= 3:
        interval = 0.5
    else:
        interval = 1
    plt.xticks(np.arange(0., x_ul+2*interval, interval), fontsize=13)
    plt.yticks(np.arange(y_ll, y_ul+1, 0.5), fontsize=13)
    plt.grid()
    plt.tick_params(which='both',
                    top=True, labeltop=False, 
                    right=True, labelright=False)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/potency_rate.png'.format(rel, fname), dpi=600)
    plt.close()



    fig, ax = plt.subplots(figsize=(6, 4))
    for EQ in range(EQ_num):
        ax.vlines(x=EQ_start[EQ]/yr2sec, 
                  ymin=np.min(np.log10(Pot)),
                  ymax=np.log10(Pot[EQ]),
                  color='dimgray', lw=1, zorder=1)
    ax.scatter(EQ_start/yr2sec,
               np.log10(Pot),
               marker='o', s=30, facecolor='black',
               edgecolor='gray', linewidth=0.5, zorder=2)
    ax.set_xlabel('Simulation time (year)', fontsize=15)
    ax.set_ylabel(r'Log potency (m$^2$/s)', fontsize=12)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tick_params(which='both',
                    top=True, labeltop=False, 
                    right=True, labelright=False)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/PT.png'.format(rel, fname), dpi=600)
    plt.close()



    fig, ax = plt.subplots(figsize=(5, 5))
    x_ll = np.min(np.log10(Pot))
    x_ul = np.max(np.log10(Pot))
    if x_ul - x_ll < 1.:
        x_ll = np.floor(x_ll)
        x_ul = np.ceil(x_ul)
        ax.hist(np.log10(Pot), range=(x_ll, x_ul),
                bins=20, facecolor='black', edgecolor='silver', lw=1.5, zorder=2)
    else:
        x_ll = np.floor(x_ll)
        x_ul = np.ceil(x_ul)
        ax.hist(np.log10(Pot), range=(x_ll, x_ul),
                bins=int(10*(x_ul-x_ll)), facecolor='black', edgecolor='silver', lw=1.5, zorder=2)
    ax.set_xlabel(r'Log potency (m$^2$/s)', fontsize=15)
    ax.set_ylabel('Event count', fontsize=15)
    ax.set_yscale('log')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.minorticks_on()
    plt.grid(zorder=1)
    
    plt.tick_params(which='both',
                    top=True, labeltop=False, 
                    right=True, labelright=False)
    
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/GR.png'.format(rel, fname), dpi=600)
    plt.close()



    fig, ax = plt.subplots(figsize=(5, 5))
    duration = EQ_end - EQ_start
    x = np.log10(Pot)
    y = np.log10(duration)
    x_ul = np.ceil(np.max(x))
    x_ll = np.floor(np.min(x))
    y_ul = np.ceil(np.max(y))
    y_ll = np.floor(np.min(y))
    ax.scatter(x, y, facecolor='None', edgecolor='black', lw=2, s=100, zorder=2)
    ax.set_xlabel(r'Log potency $\log_{10}(P)$ (m$^2$)', fontsize=15)
    ax.set_ylabel(r'Log event duration $\log_{10}(T)$ (sec)', fontsize=15)
    ax.set_xlim(x_ll-0.5, x_ul+0.5)
    ax.set_ylim(y_ll-0.5, y_ul+0.5)
    if x_ul - x_ll <= 2:
        interval = 0.5
    else:
        interval = 1
    plt.xticks(np.arange(x_ll-0.5, x_ul+0.5, interval), fontsize=13)
    if y_ul - y_ll <= 1:
        interval = 0.2
    else:
        interval = 0.5
    plt.yticks(np.arange(y_ll-0.5, y_ul+0.5, interval), fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tick_params(which='both',
                    top=True, labeltop=False, 
                    right=True, labelright=False)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/potency_duration.png'.format(rel, fname), dpi=600)
    plt.close()
    
    
    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.log10(Pot)
    y = sigma_M*1.e-6
    x_ul = np.ceil(np.max(x))
    x_ll = np.floor(np.min(x))
    y_ul = np.ceil(np.max(y))
    y_ll = np.floor(np.min(y))
    ax.scatter(x, y, facecolor='None', edgecolor='orangered', lw=2, s=100, zorder=2)
    ax.set_xlabel(r'Log potency $\log_{10}(P)$ (m$^2$)', fontsize=15)
    ax.set_ylabel(r'Stress drop $\Delta\sigma$ (MPa)', fontsize=15)
    ax.set_xlim(x_ll-0.5, x_ul+0.5)
    ax.set_ylim(y_ll-0.5, y_ul+0.5)
    if x_ul - x_ll <= 2:
        interval = 0.5
    else:
        interval = 1
    plt.xticks(np.arange(x_ll-0.5, x_ul+0.5, interval), fontsize=13)
    plt.yticks(np.arange(y_ll-0.5, y_ul+0.5, interval), fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    
    plt.tick_params(which='both',
                    top=True, labeltop=False, 
                    right=True, labelright=False)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/stress_drop.png'.format(rel, fname), dpi=600)
    plt.close()
    
    
    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.log10(Pot)
    y = np.log10(ER)
    x_ul = np.ceil(np.max(x))
    x_ll = np.floor(np.min(x))
    y_ul = np.ceil(np.max(y))
    y_ll = np.floor(np.min(y))
    ax.scatter(x, y, facecolor='None', edgecolor='lightseagreen', lw=2, s=100, zorder=2)
    ax.set_xlabel(r'Log potency $\log_{10}(P)$ (m$^2$)', fontsize=15)
    ax.set_ylabel(r'Log radiated energy $E_{\rm{R}}$ (J/m)', fontsize=15)
    ax.set_xlim(x_ll-0.5, x_ul+0.5)
    ax.set_ylim(y_ll-0.5, y_ul+0.5)
    if x_ul - x_ll <= 2:
        interval = 0.5
    else:
        interval = 1
    plt.xticks(np.arange(x_ll-0.5, x_ul+0.5, interval), fontsize=13)
    plt.yticks(np.arange(y_ll-0.5, y_ul+0.5, interval), fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tick_params(which='both',
                    top=True, labeltop=False, 
                    right=True, labelright=False)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/ER.png'.format(rel, fname), dpi=600)
    plt.close()
    
    
    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.log10(Pot)
    y = np.log10(Ea)
    x_ul = np.ceil(np.max(x))
    x_ll = np.floor(np.min(x))
    y_ul = np.ceil(np.max(y))
    y_ll = np.floor(np.min(y))
    ax.scatter(x, y, facecolor='None', edgecolor='limegreen', lw=2, s=100, zorder=2)
    ax.set_xlabel(r'Log potency $\log_{10}(P)$ (m$^2$)', fontsize=15)
    ax.set_ylabel(r'Log available energy $E_{\rm{a}}$ (J/m)', fontsize=15)
    ax.set_xlim(x_ll-0.5, x_ul+0.5)
    ax.set_ylim(y_ll-0.5, y_ul+0.5)
    if x_ul - x_ll <= 2:
        interval = 0.5
    else:
        interval = 1
    plt.xticks(np.arange(x_ll-0.5, x_ul+0.5, interval), fontsize=13)
    plt.yticks(np.arange(y_ll-0.5, y_ul+0.5, interval), fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    
    plt.tick_params(which='both',
                    top=True, labeltop=False, 
                    right=True, labelright=False)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/Ea.png'.format(rel, fname), dpi=600)
    plt.close()
    
    
    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.log10(Pot)
    y = np.log10(ER/Ea)
    x_ul = np.ceil(np.max(x))
    x_ll = np.floor(np.min(x))
    y_ul = np.ceil(np.max(y))
    y_ll = np.floor(np.min(y))
    ax.scatter(x, y, facecolor='None', edgecolor='darkviolet', lw=2, s=100, zorder=2)
    ax.set_xlabel(r'Log potency $\log_{10}(P)$ (m$^2$)', fontsize=15)
    ax.set_ylabel(r'Log radiation efficiency $\eta_{\rm{R}}$', fontsize=15)
    ax.set_xlim(x_ll-0.5, x_ul+0.5)
    ax.set_ylim(y_ll-0.5, y_ul+0.5)
    if x_ul - x_ll <= 2:
        interval = 0.5
    else:
        interval = 1
    plt.xticks(np.arange(x_ll-0.5, x_ul+0.5, interval), fontsize=13)
    plt.yticks(np.arange(y_ll-0.5, y_ul+0.5, interval), fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    
    plt.tick_params(which='both',
                    top=True, labeltop=False, 
                    right=True, labelright=False)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/etaR.png'.format(rel, fname), dpi=600)
    plt.close()


    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.log10(slip_ave)
    y = np.log10(Gc)
    x_ul = np.ceil(np.max(x))
    x_ll = np.floor(np.min(x))
    y_ul = np.ceil(np.max(y))
    y_ll = np.floor(np.min(y))
    ax.scatter(x, y, facecolor='None', lw=2, s=100, zorder=2, edgecolor='deeppink')
    ax.set_xlabel(r'Average coseismic slip $\log_{10}(\widebar{s})$ (m)', fontsize=15)
    ax.set_ylabel(r'Average fracture energy $G_{\rm{c}}^{\prime}$ (J/m$^2$)', fontsize=15)
    ax.set_xlim(x_ll-0.5, x_ul+0.5)
    ax.set_ylim(y_ll-0.5, y_ul+0.5)
    if x_ul - x_ll <= 1:
        interval = 0.2
    else:
        interval = 0.5
    plt.xticks(np.arange(x_ll-0.5, x_ul+0.5, interval), fontsize=13)
    plt.yticks(np.arange(y_ll-0.5, y_ul+0.5, interval), fontsize=13)
    plt.minorticks_on()
    plt.grid(zorder=1)
    plt.tick_params(which='both',
                    top=True, labeltop=False, 
                    right=True, labelright=False)
    plt.tight_layout()
    plt.savefig('{}Figures{}Summary/Gc_slip.png'.format(rel, fname), dpi=600)
    plt.close()

    print('Summary')