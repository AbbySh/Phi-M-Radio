import matplotlib.pyplot as plt
import pickle
import numpy as np
from alg.StellarRadio_alg import StellarRadioAlg

class StellarRadioAlg_Plotting:

    def __init__(self,instrument,id):
        self.instrument = instrument
        self.id = id

        if self.instrument == 'kepler':
            self.ins_prefix = 'kic'
        elif self.instrument == 'tess':
            self.ins_prefix = 'tic'

    def load_values(self):
        """
        Loads values from pickle file, or, performs stellar radio algorithm if no pickle file for stellar object exists. 

        Returns:
            np.ndarray: Time data.
            np.ndarray: Flux data.
            np.ndarray: Number of light curve sectors.
            float: Guess for pulsation frequency.
            np.ndarray: Frequency (x-) component of LombScargle.
            np.ndarray: Power (y-) component of LombScargle.
            np.ndarray: Optimized Gaussian-filtered immf.
        """
        try:
            with open('./pickle_files/pickle_{}_{}.pkl'.format(self.ins_prefix,self.id),'rb') as file:
                print('pickle file exists for object, plotting now')
                vals = pickle.load(file)
                time = vals['time']
                flux = vals['flux']
                modes = vals['modes']
                qs = vals['qs']
                ys = vals['ys']
                simmfs = vals['simmfs']
                q_modes = vals['q_modes']
                y_modes = vals['y_modes']
            

        except:
            print('pickle file does not exist: running phi-m script then will plot')
            alg_init = StellarRadioAlg()
            alg_init.run_all_steps()
            with open('./pickle_files/pickle_{}_{}.pkl'.format(self.ins_prefix,self.id),'rb') as file:
                vals = pickle.load(file)
                time = vals['time']
                flux = vals['flux']
                modes = vals['modes']
                qs = vals['qs']
                ys = vals['ys']
                simmfs = vals['simmfs']
                q_modes = vals['q_modes']
                y_modes = vals['y_modes']

        return time,flux,modes,qs,ys,simmfs

    def lightcurve_plot(self,time,flux):
        """
        Plots lightcurve.

        Args:
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
        """
        plt.figure()
        plt.title('{}{} Lightcurve'.format(self.ins_prefix,self.id))
        plt.scatter(time,flux,marker='.',s=5,alpha=.5)
        plt.plot(time,flux,alpha=.2)
        plt.ylabel('Flux (arbitrary units)')
        plt.xlabel('Barycentric Julian Date Plus Offset (d)')
        plt.savefig('./lightcurves/stellar_radio_lightcurve_{}_{}.pdf'.format(self.ins_prefix,str(self.id)))
        plt.show()

    # def folded_flux_plot(self,time,flux,f0):
        # t0 = 1/f0
        # folded_time = time % t0
        # means,bin_edges,_ = bs(folded_time,flux,bins=100)
        # bin_centers = .5*(bin_edges[1:]+bin_edges[:-1])

        # plt.figure()
        # plt.title('Lightcurve: {}{} Folded Flux'.format(self.ins_prefix,self.id))
        # plt.scatter(folded_time,flux,marker='.',s=5,alpha=.03)
        # plt.scatter(folded_time+t0,flux,marker='.',s=5,alpha=.03)
        # plt.plot(bin_centers,means,'ko')
        # plt.ylabel('Flux (arbitrary units)')
        # plt.xlabel('Modded Time (d)')
        # plt.savefig('./lightcurves/stellar_radio_lightcurve_{}_{}_folded.pdf'.format(self.ins_prefix,str(self.id)))
        # plt.show()

    def modes_plot(self,time,q_modes,y_modes):
        """
        Plots modes up to nyquist frequency (LombScargle).

        Args:
            time (np.ndarray): Time data.
            q (np.ndarray): Frequency (x-) component of LombScargle.
            y (np.ndarray): Power (y-) component of LombScargle.
            f0 (float): Guess for pulsation frequency.
        """
        sampling_freq = 1./np.nanmedian(time[1:]-time[:-1])
        nyquist = .5*sampling_freq
        plt.figure()
        plt.title('{}{} Modes'.format(self.ins_prefix,self.id))
        plt.plot(q_modes,y_modes,c='purple',alpha=.5)
        plt.axvline(sampling_freq,c='black',alpha=.25)
        # plt.axvline(f0,alpha=.5,color='green')
        plt.xlim(0,nyquist)
        plt.xlabel('Frequency (1/d)')
        plt.ylabel('Amplitude (? units)')
        plt.savefig('./frequency_guess_plots/mode_guess_{}_{}.pdf'.format(self.ins_prefix,self.id))
        plt.show()

    # def phase_plot(self):


    def final_lombscargle_plot(self,qs,ys,modes):
        """Sets up plot for final LombScargle periodogram.

        Args:
            qs(np.ndarray):
            ys(np.ndarray): 
            modes(np.ndarray): Array of mode values.
        """
        plt.figure()
        for no,(q,y) in enumerate(zip(qs,ys)):
            plt.plot(1./q,y,alpha=.3,marker='.',label='{}'.format(modes[no])) #this may be a bug, test
        plt.title('{}{} Periodogram of Period vs Gaussian-filtered Im(mixer * flux)'.format(self.ins_prefix,self.id))
        plt.xlabel("Period (days)")
        plt.ylabel('Amplitude (? units)')
        plt.xscale('log')
        plt.xlim(1)
        plt.ylim(0)
        # if expect_period == True:
        #     plt.axvline(self.period,c='mediumslateblue',alpha=.6,label="Expected Companion Orbital Period (lit): {}".format(self.period),zorder=-10)
        plt.legend(title = 'Modes')
        plt.savefig('./LS_periodograms/periodogram_{}_{}.pdf'.format(self.ins_prefix,str(self.id)))
        plt.show()

    def do_plots(self,lightcurve_plt=False,modes_plt=False,periodogram_plt=True):
        """
        Plots the plots.

        Args:
            lightcurve(bool): Defaults to False.
            modes(bool): Defaults to False.
            periodogram(bool): Defaults to True.
        """
        flux,time,modes,simmfs,qs,ys,q_modes,y_modes = self.load_values()
        print(qs,ys,modes)

        if lightcurve_plt == True:
            self.lightcurve_plot(time,flux)

        if modes_plt == True:
            self.modes_plot(time,q_modes,y_modes)

        if periodogram_plt == True:
            self.final_lombscargle_plot(qs,ys,modes)
