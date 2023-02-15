import matplotlib.pyplot as plt
import configparser
from astropy.timeseries import LombScargle
import pickle
import numpy as np
from scipy.stats import binned_statistic as bs
from alg.StellarRadio_alg import StellarRadioAlg

class StellarRadioAlg_Plotting:

    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read('stellar.cfg')
        self.freq_toggle = bool(parser.get('FIGS','freq_plot_toggle'))
        self.lombscargle_toggle = bool(parser.get('FIGS','lombscargle_toggle'))
        self.lightcurve_toggle = bool(parser.get('FIGS','lightcurve_toggle'))
        self.instrument = str(parser.get('STARQUALS','instrument'))
        self.id = int(parser.get('STARQUALS','id'))
        self.period = float(parser.get('FIGS','period'))

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
                print('pickle file exists for object')
                vals = pickle.load(file)
                time = vals['time']
                flux = vals['flux']
                quarter = vals['quarter']
                f0 = vals['f0']
                q = vals['q']
                y = vals['y']
                simmf = vals['simmf']
            
        except:
            print('pickle file does not exist: running necessary algorithmic steps')
            alg_init = StellarRadioAlg()
            alg_init.run_all_steps()
            with open('./pickle_files/pickle_{}_{}.pkl'.format(self.ins_prefix,self.id),'rb') as file:
                vals = pickle.load(file)
                time = vals['time']
                flux = vals['flux']
                quarter = vals['quarter']
                f0 = vals['f0']
                q = vals['q']
                y = vals['y']
                simmf = vals['simmf']

        return time,flux,quarter,f0,q,y,simmf

    def lightcurve_plot(self,time,flux,quarter,f0):
        """
        Plots lightcurve.

        Args:
            time (np.ndarray): Time data
            flux (np.ndarray): Flux data
            quarter (np.ndarray): Number of light curve sectors.
            f0 (float): Guess for pulsation frequency.
        """
        plt.figure()
        plt.title('Lightcurve: {}{} PDCSAP Flux'.format(self.ins_prefix,self.id))
        plt.scatter(time,flux,marker='.',s=5,alpha=.7)
        plt.ylabel('Flux (arbitrary units)')
        plt.xlabel('Barycentric Julian Date Plus Offset (d)')
        plt.savefig('./lightcurves/stellar_radio_lightcurve_{}_{}.pdf'.format(self.ins_prefix,str(self.id)))
        plt.show()
        plt.close()

        t0 = 1/f0
        folded_time = time % t0
        means,bin_edges,_ = bs(folded_time,flux,bins=100)
        bin_centers = .5*(bin_edges[1:]+bin_edges[:-1])

        plt.figure()
        plt.title('Lightcurve: {}{} Folded Flux'.format(self.ins_prefix,self.id))
        plt.scatter(folded_time,flux,marker='.',s=5,alpha=.03)
        plt.scatter(folded_time+t0,flux,marker='.',s=5,alpha=.03)
        plt.plot(bin_centers,means,'ko')
        plt.ylabel('Flux (arbitrary units)')
        plt.xlabel('Modded Time (d)')
        plt.savefig('./lightcurves/stellar_radio_lightcurve_{}_{}_folded.pdf'.format(self.ins_prefix,str(self.id)))
        plt.show()
        plt.close()

    def freq_guess_plot(self,time,q,y,f0):
        """
        Plots frequency guess plot (LombScargle).

        Args:
            time (np.ndarray): Time data.
            q (np.ndarray): Frequency (x-) component of LombScargle.
            y (np.ndarray): Power (y-) component of LombScargle.
            f0 (float): Guess for pulsation frequency.
        """
        sampling_freq = 1./np.nanmedian(time[1:]-time[:-1])
        plt.figure()
        plt.title('LombScargle Periodogram: {}{} Frequency Guess'.format(self.ins_prefix,self.id))
        plt.plot(q,y,c='purple',alpha=.5)
        plt.axvline(sampling_freq,c='black',alpha=.25)
        plt.axvline(f0,alpha=.5,color='green')
        plt.savefig('./frequency_guess_plots/stellar_radio_frequency_guess_{}_{}.pdf'.format(self.ins_prefix,str(self.id)))
        plt.xlabel('Frequency (1/d)')
        plt.ylabel('Measurement Values (arbitrary units)')
        #plt.xlim(0,20)
        plt.show()
        plt.close()

    def final_lombscargle_plot(self,time,simmf):
        """Sets up plot for final LombScargle periodogram.

        Args:
            time (np.ndarray): Time data
            simmf (np.ndarray): Optimized Gaussian-filtered immf
        """
        q,y = LombScargle(time,simmf).autopower()
        plt.plot(1./q,y,c='black',alpha=.75)
        plt.title('LombScargle Periodogram: {}{} Period vs Gaussian-filtered Im(mixer * flux)'.format(self.ins_prefix,self.id))
        plt.xlabel("Period (days)")
        plt.ylabel('"Simmf" (arbitrary units)')
        plt.xlim(1,100)
        plt.grid()
        plt.loglog()
        plt.axvline(self.period,c='mediumslateblue',alpha=.6,label="expected companion orbital period (literature): {}".format(self.period),zorder=-10)
        plt.legend()
        plt.savefig('./LS_periodograms/stellar_radio_plot_{}_{}.pdf'.format(self.ins_prefix,str(self.id)))
        plt.show()
        plt.close()

    def do_plots(self):
        """
        Plots the plots.
        """
        time,flux,quarter,f0,q,y,simmf = self.load_values()
        if self.lightcurve_toggle == True:
            self.lightcurve_plot(time,flux,quarter,f0)

        if self.freq_toggle == True:
            self.freq_guess_plot(time,q,y,f0)

        if self.lombscargle_toggle == True:
            self.final_lombscargle_plot(time,simmf)
