import numpy as np
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
import scipy.ndimage
import lightkurve as lk
import configparser
import pickle

class StellarRadioAlg:

    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read('stellar.cfg')
        self.instrument = str(parser.get('STARQUALS','instrument'))
        self.id = int(parser.get('STARQUALS','id'))
        self.cadence = str(parser.get('STARQUALS','cadence'))
        self.amp0 = float(parser.get('MATH','amp0'))
        self.phase0 = float(parser.get('MATH','phase0'))
        self.iters = int(parser.get('MATH','iters'))
        self.period = float(parser.get('FIGS','period'))

        if self.instrument == 'kepler':
            self.ins_prefix = 'kic'
        elif self.instrument == 'tess':
            self.ins_prefix = 'tic'

    def freq_finder(self,time,flux,qmin=0.,qmax=np.Inf):
        """Finds frequency based off of LombScargle peaks.

        Args:
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            qmin (float): Minimum for frequency choice.
            qmax (float): Maximum for frequency choice.

        Returns:
            float: Frequency estimate that is improved upon in optimization step.
            np.ndarray: Frequency (x-) component of LombScargle.
            np.ndarray: Power (y-) component of LombScargle.
        """            
        q,y = LombScargle(time,flux).autopower()
        f0_guess = q[np.argmax(y * (q > qmin) * (q < qmax))]
        print("Frequency guess:",f0_guess)

        return f0_guess,q,y

    def get_lc_data(self):
        """Gets light curve data using Lightkurve.

        Returns:
            np.ndarray: Time data.
            np.ndarray: Flux data.
            np.ndarray: Number of light curve sectors.
        """
        if type(self.id) != str:
            self.id = str(self.id)

        if self.instrument == 'tess':
            try:
                lc_files = lk.search_lightcurve('TIC'+self.id, cadence='short').download_all()
                print('cadence type: short')
            except:
                lc_files = lk.search_lightcurve('TIC'+self.id, cadence='long').download_all()
                print('cadence type: long')

        elif self.instrument == 'kepler':
            lc_files = lk.search_lightcurve('KIC'+self.id, cadence=self.cadence).download_all()
            print('cadence type: {}'.format(self.cadence))

        else:
            raise TypeError('Only Kepler and TESS are supported instruments')
            
        time, flux, quarter = np.array([]), np.array([]), np.array([])
        for q,lc in enumerate(lc_files):

            time_table = lc['time']
            this_time = []
            for val in range(0,len(time_table)):
                time_value = time_table[val].value
                this_time.append(time_value)

            this_flux = lc['pdcsap_flux']
            good = np.isfinite(this_time)
            
            median_flux = np.nanmedian(this_flux)
            this_flux = this_flux[good] / median_flux
            this_q = np.zeros_like(this_time) + q
            
            bad = np.logical_not(np.isfinite(this_flux))
            this_flux[bad] = 1.
            time = np.concatenate((time,this_time))
            flux = np.array(np.concatenate((flux,this_flux)))
            quarter = np.concatenate((quarter, this_q))

            ###mask hack###
            t0 = 0
            per = 58
            for i in np.linspace(0,100):
                event_time = t0 + (i*per)
                mask = (time > (event_time - .5)) & (time < (event_time + .5))
                time = time[~mask]
                flux = flux[~mask]

        quarter = np.round(quarter).astype(int)

        flux=flux-1.

        return time,flux,quarter

    def eclipse_mask(self):
        """
        Masks out eclipses from light curve.
        """
        pass


    def mix(self,time,flux,f0):
        """Out-of-phase carrier wave multiplication with flux data.

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
        print('mix:',self.amp0,f0,self.phase0,np.median(flux),np.min(flux),np.mean(flux),np.mean(mf))
        return sremf,simmf

    def objective(self,f,time,flux,amp,phase):
        """Gets variance of imaginary component of the mixer multiplied by flux.

        Args:
            f (float): Frequency value.
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
            np.ndarray: Optimized Gaussian-filtered immf.
        """
        for i in range(self.iters):
            print("Iteration", i)
            
            sremf,simmf = self.mix(time,flux,f0)

            new_amp = self.amp0 / np.sqrt(np.mean(sremf**2 + simmf**2))
            print(np.mean(sremf**2 + simmf**2),np.mean(sremf**2),np.mean(simmf**2))
            print("Amps:", self.amp0, new_amp)
            self.amp0 = new_amp
            
            sremf,simmf = self.mix(time,flux,f0)
            new_phase = self.phase0 + np.arctan2(np.mean(simmf), np.mean(sremf))
            print("Phases:", self.phase0, new_phase)
            self.phase0 = new_phase
        
            sremf, simmf = self.mix(time,flux,f0)

            output=minimize(self.objective,f0,args=(time,flux,self.amp0,self.phase0),method="Nelder-Mead")
            print("Frequencies:", f0, np.mean((output.final_simplex[0][0],output.final_simplex[0][1])))
            f0=np.mean((output.final_simplex[0][0],output.final_simplex[0][1]))
            
        return simmf

    def run_all_steps(self, injection_flux=None,qmin=0,qmax=np.Inf): # flux hack! Instead this should take injection parameters.
        """Runs algorithm steps in order, which ends in plotting a LombScargle
        periodogram of time vs simmf.

        Args:
            injection_flux (np.ndarray): Flux to be injected into stellar object flux. Used for testing. Defaults to None.
            qmin (float): Minimum boundary for frequency guess. Defaults to 0.
            qmax (float): Maximum boundary for frequency guess. Defaults to infinity.

        Returns:
            np.ndarray: Time data.
            np.ndarray: Flux data.
            np.ndarray: Number of light curve sectors.
            np.ndarray: Optimized Gaussian-filtered immf.
        """
        time,flux,quarter = self.get_lc_data()

        if injection_flux is not None:
            flux = injection_flux

        f0,q,y = self.freq_finder(time,flux,qmin=qmin,qmax=qmax)

        simmf = self.optimization(time,flux,f0)

        plot_vals = {'flux':flux,'time':time,'quarter':quarter,'f0':f0,'q':q,'y':y,'simmf':simmf}

        with open('./pickle_files/pickle_{}_{}.pkl'.format(self.ins_prefix,self.id),'wb') as file:
            pickle.dump(plot_vals,file)
            file.close()
        print ('done')
        return time,flux,quarter,simmf
