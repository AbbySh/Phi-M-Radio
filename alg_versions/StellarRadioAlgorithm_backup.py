import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
import scipy.ndimage
from scipy.fftpack import fft,ifft,fftfreq
import finufft
import lightkurve as lk
import configparser

class StellarRadioAlg:

    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read('stellar.cfg')
        self.kics = parser.get('STARQUALS','kics')
        self.bandwidth = float(parser.get('MATH','bandwidth'))
        self.amp0 = float(parser.get('MATH','amp0'))
        self.phase0 = float(parser.get('MATH','phase0'))
        self.iters = int(parser.get('MATH','iters'))
        self.periods = parser.get('FIGS','periods')
        self.period_alphap = float(parser.get('FIGS','period_alphap'))
        self.ymin = float(parser.get('FIGS','ymin'))
        self.ymax = float(parser.get('FIGS','ymax'))
        self.xmin = float(parser.get('FIGS','xmin'))
        self.xmax = float(parser.get('FIGS','xmax'))
        self.debug_plots = parser.get('FIGS','debug_plots')

    def kic_loop(self):
        """Takes KIC list and period list from config, turns them into integers 
        to be looped over.
        """
        kics_split = self.kics.split(',')
        kic_list = []
        for kic in kics_split:
            kic_list.append(int(kic))
        
        period_split = self.periods.split(',')
        period_list = []
        for period in period_split:
            period_list.append(float(period))

        return kic_list,period_list

    def freq_finder(self,kic):
        """Finds frequency based off of LombScargle peaks.

        Returns:
            f0_guess (float): Frequency estimate that is improved upon in optimization step.
        """
        time, flux, flerr, quarter = self.get_lc_data(kic)
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

    def get_lc_data(self,kic):
        """Gets light curve data using Lightkurve.

        Returns:
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            flerr (np.ndarray): Flux-error data.
            quarter (np.ndarray): Number of light curves.
        """
        if type(kic) == str:
            lc_files = lk.search_lightcurve('KIC'+kic, cadence='long').download_all()
        else:
            lc_files = lk.search_lightcurve('KIC'+str(kic), cadence='long').download_all()

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
            flux (np.ndarray): Bandpass-filtered flux data.
        """
        q_sort = np.unique(np.sort(quarters))
        altered_flux = flux
        for quarter in q_sort:
            I = quarters == quarter
            
            _flux = altered_flux[I]
            this_time = time[I]
            fft_flux = fft(_flux)
            fftfreq_time = fftfreq(len(_flux)) / np.nanmedian(
                this_time[1:] - this_time[:-1])
            _filter = np.abs(np.abs(fftfreq_time) - f0) > 3 * self.bandwidth
            fft_flux[_filter] = 0.
            
            this_flux = ifft(fft_flux).real
            altered_flux[I] = this_flux
        return altered_flux

    def mix(self,time,flux,f0):
        """Mix flux. ? Bad description

        Args:
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            f0 (float): Frequency guess.

        Returns:
            remf (np.ndarray): Real components of mixer multiplied by flux.
            immf (np.ndarray): Imaginary components of mixer multiplied by flux.
            sremf (np.ndarray): Gaussian-filtered remf.
            simmf (np.ndarray): Gaussian-filtered immf.
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
            f (float): Frequency.
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            amp (float): Amplitude value.
            phase (float): Phase value.

        Returns:
            np.var(mf.imag) (np.ndarray): Variance of imaginary component of mixer 
                multiplied by float.
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
            remf (np.ndarray): Optimized real components of mixer multiplied by flux.
            immf (np.ndarray): Optimized imaginary components of mixer multiplied by flux.
            sremf (np.ndarray): Optimized Gaussian-filtered remf.
            simmf (np.ndarray): Optimized Gaussian-filtered immf.
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

            output=minimize(self.objective,f0,args=(
                time,flux,self.amp0,self.phase0),
                method="Nelder-Mead")
            print("Frequencies:", f0, np.mean((output.final_simplex[0][0],output.final_simplex[0][1])))
            f0=np.mean((output.final_simplex[0][0],output.final_simplex[0][1]))
            
        return remf, immf, sremf, simmf,output

    def lombscargle_periodogram(self,time,simmf,kic,period):
        """Sets up and plots final LombScargle periodogram.

        Args:
            time (np.ndarray): Time data.
            simmf (np.ndarray): Optimized Gaussian-filtered immf.
        """
        plt.figure()
        plt.title('Lombscargle Periodogram of KIC{}'.format(kic))
        q,y = LombScargle(time,simmf).autopower()
        plt.plot(1./q,y)
        plt.xlim((self.xmin,self.xmax))
        plt.ylim((self.ymin,self.ymax))
        plt.axvline(372.5,c='r',alpha=self.period_alphap,label="Kepler orbital period (372.5d)")
        plt.axvline(period,c='g',alpha=self.period_alphap,label="Expected frequency peak: {} d".format(period))
        plt.xlabel("Period (days)")
        plt.ylabel("LombScargle")
        plt.legend()
        plt.grid()
        plt.loglog()
        plt.savefig('./periodograms/LS_periodogram_kic{}.pdf'.format(kic))
        # if self.debug_plots == 'True':
        #     plt.figure()
        #     plt.title('Lombscargle Periodogram of KIC{}, Zoomed'.format(kic))
        #     plt.plot(1./q,y)
        #     plt.xlim((self.xmin,self.xmax))
        #     plt.ylim((self.ymin,self.ymax))

    def debug_plots_fxn(self,time,flux,new_flux,quarter,kic):
        """When toggled, will create plots to use for debugging.

        Args:
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            new_flux (np.ndarray): Bandpass-filtered flux data.
            quarter ([type]): ?
            kic (int): KIC ID number.
        """
        plt.figure()
        plt.scatter(time,flux,marker='.',alpha=self.period_alphap,c=quarter)
        plt.title('Flux vs Time')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.savefig('./debug_plots/fluxvtime_kic{}.pdf'.format(kic))
        plt.close()

        plt.figure()
        plt.scatter(time,new_flux,marker='.',alpha=self.period_alphap,c=quarter)
        plt.title('Bandpass-filtered Flux vs Time')
        plt.xlabel('Time')
        plt.ylabel('Bandpass-Filtered Flux')
        plt.savefig('./debug_plots/bpass_fluxvtime_kic{}.pdf'.format(kic))
        plt.close()

    def run_all_steps(self):
        """Runs algorithm steps in order, which ends in plotting a LombScargle
        periodogram of time vs simmf.
        """
        kic_list,period_list = self.kic_loop()
        for kic,period in zip(kic_list,period_list):
            time,flux,flerr,quarter = self.get_lc_data(kic)
            if self.debug_plots == 'True':
                plt.figure()
                plt.scatter(time,flux,marker='.',alpha=self.period_alphap,c=quarter)
                plt.title('Flux vs Time')
                plt.xlabel('Time')
                plt.ylabel('Flux')
                plt.savefig('./debug_plots/fluxvtime_kic{}.pdf'.format(kic))
                plt.close()
            f0 = self.freq_finder(kic)
            new_flux = self.bandpass_filter(flux,time,quarter,f0)
            if self.debug_plots == 'True':
                plt.figure()
                plt.scatter(time,new_flux,marker='.',alpha=self.period_alphap,c=quarter)
                plt.title('Bandpass-filtered Flux vs Time')
                plt.xlabel('Time')
                plt.ylabel('Bandpass-Filtered Flux')
                plt.savefig('./debug_plots/bpass_fluxvtime_kic{}.pdf'.format(kic))
                plt.close()
            remf, immf, sremf, simmf, oput = self.optimization(time,new_flux,f0)
            self.lombscargle_periodogram(time,simmf,kic,period)
            return(time,flux)
