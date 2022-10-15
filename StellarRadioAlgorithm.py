import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
import scipy.ndimage
from scipy.fftpack import fft,ifft,fftfreq
import lightkurve as lk
import configparser

class StellarRadioAlg:

    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read('stellar.cfg')
        self.kic = int(parser.get('STARQUALS','kic'))
        self.bandwidth = float(parser.get('MATH','bandwidth'))
        self.amp0 = float(parser.get('MATH','amp0'))
        self.phase0 = float(parser.get('MATH','phase0'))
        self.iters = int(parser.get('MATH','iters'))
        self.period = float(parser.get('FIGS','period'))

        # self.alphap = parser.get('FIGS','alphap')
        # self.datac = parser.get('FIGS','datac')
        # self.resultc = parser.get('FIGS','resultc')
        # self.guidingc = parser.get('FIGS','guidingc')
        # self.axisfsize = parser.get('FIGS','axisfsize')
        # self.figsizerect = parser.get('FIGS','figsizerect')
        # self.figsizesq = parser.get('FIGS','figsizesq')
        # self.ylim = parser.get('FIGS','ylim')
        # self.xlim = parser.get('FIGS','xlim')

    def freq_finder(self):
        """Finds frequency based off of LombScargle peaks.

        Returns:
            float: Frequency estimate that is improved upon in optimization step.
        """
        time, flux, flerr, quarter = self.get_lc_data()
        for item in (time,flux,flerr,quarter):
            if not np.all(np.isfinite(item)):
                bad = np.logical_not(np.isfinite(item))
                print(np.sum(bad))
        I = (quarter < 100)
        q,y = LombScargle(time[I],flux[I]).autopower()
        firsthalf_max = list(y).index(max(y[0:int(len(y)/2)]))
        f0_guess = q[firsthalf_max]
        print("Frequency guess:",f0_guess)
        return f0_guess

    def get_lc_data(self):
        """Gets light curve data using Lightkurve.

        Returns:
            np.ndarray: Time data.
            np.ndarray: Flux data.
            np.ndarray: Flux-error data.
            np.ndarray: Number of light curves.
        """
        if type(self.kic) == str:
            lc_files = lk.search_lightcurve('KIC'+self.kic, cadence='long').download_all()
            #print(lc_files)
        else:
            lc_files = lk.search_lightcurve('KIC'+str(self.kic), cadence='long').download_all()
            #print(lc_files)

        short_curves, long_curves = [], []
        for lc in lc_files:
            if "llc" in lc.filename:
                long_curves.append(lc)
            else:
                short_curves.append(lc)
            
        time, flux, flerr, quarter = np.array([]), np.array([]), np.array([]), np.array([])
        for q,lc in enumerate(long_curves):

            time_table = lc['time']
            this_time = []
            for val in range(0,len(time_table)):
                time_value = time_table[val].value
                this_time.append(time_value)

            this_flux = lc['pdcsap_flux']
            good = np.isfinite(this_time)
            
            median_flux = np.nanmedian(this_flux)
            this_flux = this_flux[good] / median_flux
            this_flerr = lc['pdcsap_flux_err'][good] / median_flux
            this_q = np.zeros_like(this_time) + q
            
            bad = np.logical_not(np.isfinite(this_flux))
            this_flux[bad] = 1.
            this_flerr[bad] = 1000.
            time = np.concatenate((time,this_time))
            flux = np.array(np.concatenate((flux,this_flux)))
            flerr = np.array(np.concatenate((flerr,this_flerr)))
            quarter = np.concatenate((quarter, this_q))
        quarter = np.round(quarter).astype(int)
        return time,flux,flerr,quarter

    def bandpass_filter(self,flux,time,quarters,f0):
        """Performs bandpass filter on flux data.

        Args:
            flux (np.ndarray): Flux data.
            time (np.ndarray): Time data.
            quarters (np.ndarray): Quarter data. ?? Bad description
            f0 (float): Non-optimized frequency guess.

        Returns:
            np.ndarray: Bandpass-filtered flux data.
        """
        q_sort = np.unique(np.sort(quarters))
        for quarter in q_sort:
            I = quarters == quarter
            
            _flux = flux[I]
            this_time = time[I]
            fft_flux = fft(_flux)
            fftfreq_time = fftfreq(len(_flux)) / np.nanmedian(this_time[1:] - this_time[:-1])
            _filter = np.abs(np.abs(fftfreq_time) - f0) > 3 * self.bandwidth
            fft_flux[_filter] = 0.
            
            this_flux = ifft(fft_flux).real
            flux[I] = this_flux
            plt.figure()
            plt.plot(flux)
            plt.savefig(f'fft_plots/fft_plot_{quarter}.png' )
            plt.close()
        return flux

    def mix(self,time,flux,f0):
        """Mix flux. ? Bad description

        Args:
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            f0 (float): Frequency guess.

        Returns:
            np.ndarray: Real components of mixer multiplied by flux.
            np.ndarray: Imaginary components of mixer multiplied by flux.
            np.ndarray: Gaussian-filtered remf.
            np.ndarray: Gaussian-filtered immf.
        """
        mixer = self.amp0 * np.exp(1j * (2 * np.pi * f0 * time - self.phase0))
        mf = mixer * flux
        remf = mf.real
        immf = mf.imag
        sremf = scipy.ndimage.filters.gaussian_filter1d(remf, 50)
        simmf = scipy.ndimage.filters.gaussian_filter1d(immf, 50)
        return remf,immf,sremf,simmf

    def objective(self,f,time,flux,amp,phase):
        """Gets variance of imaginary component of the mixer multiplied by flux.

        Args:
            f (float): Frequency
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            amp (float): Amplitude value.
            phase (float): Phase value.

        Returns:
            np.ndarray: Variance of imaginary component of mixer multiplied by float.
        """
        mixer = amp * np.exp(1j * (2 * np.pi * f * time - phase))
        mf = mixer * flux
        return np.var(mf.imag)

    def optimization(self,time,flux,f0):
        """Frequency optimization iterations.

        Args:
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            f0 (float): Frequency guess.

        Returns:
            np.ndarray: Optimized real components of mixer multiplied by flux.
            np.ndarray: Optimized imaginary components of mixer multiplied by flux.
            np.ndarray: Optimized Gaussian-filtered remf.
            np.ndarray: Optimized Gaussian-filtered immf.
            output (?): Output of Nelder-Mead minimization.
        """
        for i in range(self.iters):
            print("Iteration", i)
            
            remf,immf,sremf,simmf = self.mix(time,flux,f0)
            new_amp = self.amp0 / np.sqrt(np.mean(sremf**2 + simmf**2))
            print("Amps:", self.amp0, new_amp)
            self.amp0 = new_amp
            
            remf,immf,sremf,simmf = self.mix(time,flux,f0)
            new_phase = self.phase0 + np.arctan2(np.mean(simmf), np.mean(sremf))
            print("Phases:", self.phase0, new_phase)
            self.phase0 = new_phase
        
            remf, immf, sremf, simmf = self.mix(time,flux,f0)

            output=minimize(self.objective,f0,args=(time,flux,self.amp0,self.phase0),method="Nelder-Mead")
            print("Frequencies:", f0, np.mean((output.final_simplex[0][0],output.final_simplex[0][1])))
            f0=np.mean((output.final_simplex[0][0],output.final_simplex[0][1]))
            
        return remf, immf, sremf, simmf,output

    def lombscargle_periodogram(self,time,simmf):
        """Sets up plot for final LombScargle periodogram.

        Args:
            time (np.ndarray): [description]
            simmf (np.ndarray): Optimized Gaussian-filtered immf.

        """
        q,y = LombScargle(time,simmf).autopower()
        plt.plot(1./q,y)
        plt.xlabel("Period (days)")
        plt.ylabel("LombScargle")
        plt.grid()
        plt.loglog()

    def run_all_steps(self):
        """Runs algorithm steps in order, which ends in plotting a LombScargle
        periodogram of time vs simmf.
        """
        time,flux,flerr,quarter = self.get_lc_data()

        f0 = self.freq_finder()

        new_flux = self.bandpass_filter(flux,time,quarter,f0)

        remf, immf, sremf, simmf, oput = self.optimization(time,new_flux,f0)

        plt.figure()
        plt.title('lombscargle periodogram')
        self.lombscargle_periodogram(time,simmf)
        plt.xlim(3,1000)
        plt.ylim(1e-3, 1e-1)
        plt.axvline(372.5,c='r',alpha=.5,label="kepler orbital period")
        plt.axvline(self.period,c='g',alpha=.5,label="expected frequency peak")
        plt.legend()
        plt.savefig('./stellar_radio_plot_kic{}.pdf'.format(self.kic))
