import sbiem_modules.Load_pickle as load
import torch as tr
import numpy as np
import pickle

fname = load.load('fname.pkl')['fname']
Conditions = load.load('Output{}Conditions.pkl'.format(fname))
Medium = load.load('Output{}Medium.pkl'.format(fname))
FieldVariables = load.load('Output{}FieldVariables.pkl'.format(fname))
FaultParams = load.load('Output{}FaultParams.pkl'.format(fname))
Devices = load.load('Output{}Devices.pkl'.format(fname))

class Vis():
    def __init__(self, Conditions=Conditions, Medium=Medium, 
                 FieldVariables=FieldVariables, FaultParams=FaultParams,
                 Devices=Devices):
        self.xp = tr.tensor(Medium['xp'], dtype=tr.float64)
        self.k_size = FaultParams['k_size']
        self.k = FaultParams['k']
        self.D_inv = FaultParams['D_inv']
        self.dim = FaultParams['dim']
        self.Lf = FaultParams['Lf']
        self.Ybar = FaultParams['Ybar']
        self.alpha = FaultParams['alpha']
        self.beta = FaultParams['beta']
        self.a = FaultParams['a']
        self.c = FaultParams['c']
        self.sigma = FaultParams['sigma']
        self.A = self.a * self.sigma
        self.C = self.c * self.sigma
        self.tauc = FaultParams['tauc']
        self.Vc = FaultParams['Vc']
        
        self.V = FieldVariables['V']
        self.state = FieldVariables['state']
        self.delta = FieldVariables['delta']
        self.tau = FieldVariables['tau']
        
        self.ncell = Medium['ncell']
        self.hcell = Medium['hcell']
        self.Vpl = Medium['Vpl']
        self.V0 = Medium['V0']
        self.Cutoff = Conditions['Cutoff']
        self.stations_uni = Conditions['stations_uni']
        self.stations = Conditions['stations']
        
        self.Devices = Devices
        
        self.a_dyn = FaultParams['a_dyn']
        self.A_dyn = self.a_dyn * self.sigma
        self.Vf = FaultParams['Vf']
        self.exponent = FaultParams['exponent']
        
        self.V_plot = np.linspace(-15, 0, 100)
        self.V_plot = tr.as_tensor(self.V_plot, dtype=tr.float64, device='cpu')
        
        self.Phi = tr.zeros((self.ncell, len(self.V_plot)), dtype=tr.float64, device='cpu')
        for i in range(self.ncell):
            for j in range(len(self.V_plot)):
                self.Phi[i, j] = self.tauc[i] + self.C[i] * tr.sqrt(tr.sum(
                    (self.k**2)*((( self.beta[i]*self.k*self.Ybar )/( self.alpha[i]*(10**self.V_plot[j]) + self.beta[i]*self.k ))**2)
                    ))
      
        self.Yss = ( ( self.beta[int(self.ncell/2)]*self.k*self.Ybar )/
            ( self.alpha[int(self.ncell/2)]*self.Vpl + self.beta[int(self.ncell/2)]*self.k )
            )
        
        
        self.tau_ss = tr.zeros((self.ncell, len(self.V_plot)), dtype=tr.float64, device='cpu')
        if not self.Cutoff:
            for i in range(self.ncell):
                for j in range(len(self.V_plot)):
                    self.tau_ss[i, j] = ((self.A_dyn[i] + (self.A[i] - self.A_dyn[i]) / (1 + (10**self.V_plot[j]/self.Vf)**0.3)) 
                                         * tr.log(10**self.V_plot[j]/self.V0) + self.Phi[i, j])
                    #self.tau_ss[i, j] = self.A[i] * tr.log(10**self.V_plot[j]/(10**self.V_plot[j] + self.Vf))+ self.A_dyn[i] * tr.log(10**self.#V_plot[j] + 10**0) + self.Phi[i, j]
                    #if 10**self.V_plot[j] <= self.inflection:
                    #    self.tau_ss[i, j] = (self.A[i])*tr.log((10**self.V_plot[j])/self.V0) + self.Phi[i, j]
                    #else:
                    #    self.tau_ss[i, j] = (self.A[i] + self.A_dyn[i])*tr.log((10**self.V_plot[j])/self.V0) + self.Phi[i, j]
        
        else:
            for i in range(self.ncell):
                for j in range(len(self.V_plot)):
                    self.tau_ss[i, j] = self.A[i]*tr.log((10**self.V_plot[j])/(10**self.V_plot[j] + self.Vc[i])) + self.Phi[i, j]

    def figs(self):
        import matplotlib.pyplot as plt
        
        self.Y_candela = np.sqrt((10**(-4))*(self.k**(-3))*(8*(np.pi**3)))
        self.Y_ohnaka = np.sqrt((10**(-3))*(self.k**(-3))*(8*(np.pi**3)))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.k, 2*self.Lf*(self.Ybar**2), lw=4, color='mediumorchid', label=r'$2L_{\rm{f}}|\widebar{Y}|^2$')
        ax.plot(self.k, 2*self.Lf*(self.Yss**2), lw=4, color='black', label=r'$2L_{\rm{f}}|Y_{\rm{ss}}|^2 (V_{\rm{ss}}=3\times10^{-9} {\rm{m/s}})$')
        ax.plot(self.k, self.Y_candela**2, lw=4, color='silver', label='Candela et al. (2012)', linestyle='dotted')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Angular wave number (m$^{-1}$)', fontsize=15)
        ax.set_ylabel('Power spectral density (m$^3$)', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()

        ax_ = ax.twiny()
        ax_.plot(2*np.pi/self.k, self.Y_ohnaka**2, lw=4, color='lightslategrey', label='Ohnaka (2003)', linestyle='dashed')
        ax_.set_xscale('log')
        ax_.set_yscale('log')
        ax_.set_xlabel('Wave length (m)', fontsize=15)
        ax_.invert_xaxis()
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc='upper right', fontsize=10, framealpha=1)
        plt.tight_layout()
        plt.savefig('Figures{}Initial/defYbar.png'.format(fname), dpi=600)


        fig = plt.figure(figsize=(8, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(self.xp, self.V, color='blue', lw=3)
        ax1.set_ylabel('Fault slip velocity (m/s)', fontsize=10)
        ax1.set_yscale('log')
        plt.grid()
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(self.xp, self.state[:, 0], color='blue', lw=1.5)
        ax2.plot(self.xp, self.state[:, int(self.k_size/4)], color='blue', lw=1.5)
        ax2.plot(self.xp, self.state[:, int(self.k_size/2)], color='blue', lw=1.5)
        ax2.plot(self.xp, self.state[:, int(3*self.k_size/4)], color='blue', lw=1.5)
        ax2.plot(self.xp, self.state[:, -1], color='blue', lw=1.5)
        #ax2.plot(self.xp, self.state[:, 1499], color='blue', lw=1.5)
        ax2.set_xlabel('Distance along fault (m)', fontsize=10)
        ax2.set_ylabel('State variable', fontsize=10)
        plt.grid()
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(self.xp, self.delta, color='blue', lw=3)
        ax3.set_ylabel('Displacement (m)', fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Initial_condistions.png'.format(fname), dpi=600)

        self.cen = int(self.ncell / 2)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.V_plot, self.Phi[self.cen]*(10**(-6)), color='deepskyblue', lw=4, label=r'$\Phi_{\rm{ss}}$')
        ax.plot(self.V_plot, self.tau_ss[self.cen]*(10**(-6)), color='darkorange', lw=4, label=r'$\tau_{\rm{ss}}$')
        ax.set_xlabel(r'$\log_{10}(V)$', fontsize=15)
        ax.set_ylabel('Shear stress (MPa)', fontsize=15)
        #ax.set_xscale('log')
        plt.xticks([-14, -10, -6, -2], fontsize=10)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15, framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Tau_ss_center.png'.format(fname), dpi=600)


        fig, ax = plt.subplots(figsize=(5, 2))
        ax.plot(self.xp, self.a, color='tomato', lw=3, marker='x')
        ax.set_xlabel('Distance along fault (m)', fontsize=10)
        ax.set_ylabel(r'$a$', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/a.png'.format(fname), dpi=600)
        
        
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.plot(self.xp, self.c, color='royalblue', lw=3)
        ax.set_xlabel('Distance along fault (m)', fontsize=10)
        ax.set_ylabel(r'$c$', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/c.png'.format(fname), dpi=600)
        
        
        self.X, self.Y = np.meshgrid(self.xp, self.V_plot)
        ax = plt.figure(figsize=(7, 5)).add_subplot(projection='3d')
        # Plot the 3D surface
        ax.plot_wireframe(self.X, self.Y, self.tau_ss.T*(10**(-6)), edgecolor='darkorange', lw=0.8, rstride=7, cstride=25)
        ax.set_xlabel('Distance along fault (m)', fontsize=13, labelpad=10)
        ax.set_ylabel(r'$\log_{10}(V)$', fontsize=13, labelpad=10)
        ax.set_zlabel(r'$\tau_{\rm{ss}}$ (MPa)', fontsize=13, labelpad=10)
        plt.xticks(fontsize=13)
        plt.yticks([-14, -10, -6, -2], fontsize=13)
        ax.tick_params(axis='z', labelsize=13)
        ax.view_init(azim=-30, elev=30)
        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        #ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
        #ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
        #ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

        #ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
        #       xlabel='X', ylabel='Y', zlabel='Z')
        plt.savefig('Figures{}Initial/Tau_ss.png'.format(fname), dpi=600)
        
        
        fig, ax = plt.subplots(figsize=(5, 2))
        #ax.axhline(0, color='dimgray', linestyle='dashed', lw=2.)
        ax.plot(self.xp, self.sigma*(10**(-6)), color='black', lw=3, label='Normal stress')
        ax.plot(self.xp, self.tau*(10**(-6)), color='black', lw=3, linestyle='dashed', label='Initial shear stress')
        ax.set_xlabel('Dstance along fault (m)', fontsize=10)
        ax.set_ylabel('Stress (MPa)', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=10, framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Initial_stress.png'.format(fname), dpi=600)

        if self.stations_uni:
            self.ch_id = (self.ncell*np.arange(0, self.stations+1, 1)/(self.stations)).astype(np.int64)
            self.ch_id[1:] = self.ch_id[1:] - 1
        else:
            self.ch_id = self.stations

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.array([self.xp[0], self.xp[-1]]), np.zeros(2, dtype=np.float64), color='black', lw=2)
        ax.scatter(self.xp[self.ch_id], np.zeros_like(self.xp[self.ch_id]), facecolor='None', edgecolor='black', s=20)
        ax.set_xlabel('Distance along fault (m)', fontsize=10)
        ax.set_title('Station location', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig('Figures{}Initial/Locations.png'.format(fname), dpi=600)