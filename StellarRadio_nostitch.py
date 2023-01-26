import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
import scipy.ndimage
from scipy.signal import find_peaks as fp
from scipy.fftpack import fft,ifft,fftfreq
from scipy.stats import binned_statistic as bs
import lightkurve as lk
import configparser

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
        sampling_freq = 1./np.nanmedian(time[1:]-time[:-1])

        plt.figure()
        plt.title('{} {} PDCSAP Flux Lightcurve'.format(self.instrument,self.id))
        plt.scatter(time,flux,c=quarter,marker='.',s=5,alpha=.7)
        plt.ylabel('Flux (arbitrary units)')
        plt.xlabel('Barycentric Julian Date Plus Offset (d)')
        plt.savefig('./stellar_radio_lightcurve_{}_{}.pdf'.format(self.instrument,str(self.id)))

        for item in (time,flux,flerr,quarter):
            if not np.all(np.isfinite(item)):
                bad = np.logical_not(np.isfinite(item))
                print(np.sum(bad))
        I = (quarter < 100)
        q,y = LombScargle(time[I],flux[I]).autopower()
        ##testing##
        new_q = []
        q_idx = []
        for i,que in enumerate(q):
            if que > 0:
                new_q.append(que)
                q_idx.append(i)
        # print(q_idx,new_q)
        firsthalf_max = list(y).index(max(y[q_idx[0]:-1]))
        #firsthalf_max = list(y).index(max(y[1000:int(len(y)/2)])) #this needs to be altered

        ##switch this to argmax this is ugly

        f0_guess = q[firsthalf_max]
        plt.figure()
        plt.title('{} {} LombScargle for Frequency Guess'.format(self.instrument,self.id))
        plt.plot(q,y,c='purple',alpha=.5)
        plt.axvline(sampling_freq,c='black',alpha=.25)
        plt.axvline(f0_guess,alpha=.5,color='green')
        plt.savefig('./stellar_radio_frequency_guess_{}_{}'.format(self.instrument,str(self.id)))

        print("Frequency guess:",f0_guess)
        ##hacking##
        t0 = 1/f0_guess
        folded_time = time % t0
        means,bin_edges,_ = bs(folded_time,flux,bins=100)
        bin_centers = .5*(bin_edges[1:]+bin_edges[:-1])

        plt.figure()
        plt.title('{} {} Folded Flux Lightcurve'.format(self.instrument,self.id))
        plt.scatter(folded_time,flux,c=quarter,marker='.',s=5,alpha=.03)
        plt.scatter(folded_time+t0,flux,c=quarter,marker='.',s=5,alpha=.03)
        plt.plot(bin_centers,means,'ko')
        plt.ylabel('Flux (arbitrary units)')
        plt.xlabel('Modded Time (d)')
        plt.savefig('./stellar_radio_lightcurve_{}_{}_folded.pdf'.format(self.instrument,str(self.id)))

        return f0_guess

    def get_lc_data(self):
        """Gets light curve data using Lightkurve.

        Returns:
            np.ndarray: Time data.
            np.ndarray: Flux data.
            np.ndarray: Flux-error data.
            np.ndarray: Number of light curves.
        """
        if type(self.id) != str:
            self.id = str(self.id)

        if self.instrument == 'tess':
            try:
                lc_files = lk.search_lightcurve('TIC'+self.id, cadence='short').download_all()
            except:
                lc_files = lk.search_lightcurve('TIC'+self.id, cadence='long').download_all()
        elif self.instrument == 'kepler':
            lc_files = lk.search_lightcurve('KIC'+self.id, cadence=self.cadence).download_all()
        else:
            raise TypeError('Only Kepler and TESS are supported instruments')
            
        time, flux, flerr, quarter = np.array([]), np.array([]), np.array([]), np.array([])
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

        flux=flux-1.

        return time,flux,flerr,quarter

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
        print('mix:',self.amp0,f0,self.phase0,np.median(flux),np.min(flux),np.mean(flux),np.mean(mf))
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
            print(np.mean(sremf**2 + simmf**2),np.mean(sremf**2),np.mean(simmf**2))
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
        plt.plot(1./q,y,c='black',alpha=.75)
        plt.xlabel("Period (days)")
        plt.ylabel("LombScargle")
        plt.xlim(1,100)
        plt.grid()
        plt.loglog()
        #p1 = self.peak_finder(q,y)
        plt.axvline(self.period,c='mediumslateblue',alpha=.6,label="expected companion orbital period (literature): {}".format(self.period),zorder=-10)
        # plt.axvline(self.period*2,c='mediumslateblue',alpha=.6,label="expected companion orbital period (literature): {}".format(self.period*2))
        # plt.axvline(self.period*3,c='mediumslateblue',alpha=.6,label="expected companion orbital period (literature): {}".format(self.period*3))
        #plt.axvline(p1,c='limegreen',label="expected companion orbital period (us): {}".format(p1))
        # plt.axvline(372.5,c='r',alpha=.5,label="kepler orbital period")

    def peak_finder(self, q, y):
        """Finds peaks in final LombScargle periodogram, theoretically corresponding to the 
        orbital period(s) (in days) of companion(s) to object of interest.

        Args:

        Returns:
            np.float: Highest peak in periodogram, most likely orbital period
        """
        good_indeces = []
        new_x = []
        q = 1./q
        for no,i in enumerate(q):
            if i >= 1 and i <= 20: #between 1 and 20 day periods for now?
                new_x.append(i)
                good_indeces.append(no)
        # new_x = q[good_indeces]
        new_y = y[good_indeces]
        peak_ys = fp(new_y)[0]
        for i in peak_ys:
            print(new_x[i],new_y[i])
        peak_y_idx = np.argmax(new_y[peak_ys])
        peak_x = new_x[peak_y_idx]
        peak_y = new_y[peak_y_idx]

        #1. ignore any xs less than 1 // 2. cut off the appropriate ys // 3. find peak of ys // 4. find x that matches that y 

        # peak_indeces = np.array(fp(new_y)[0])
        # peak_xs = []

        # for i in peak_indeces:
        #     peak_xs.append(new_q[i])

        # peak_ys = []
        # for i in peak_indeces:
        #     peak_ys.append(new_y[i])
        # peak_ys = sorted(peak_ys)

        return peak_x

    def run_all_steps(self):
        """Runs algorithm steps in order, which ends in plotting a LombScargle
        periodogram of time vs simmf.
        """
        time,flux,flerr,quarter = self.get_lc_data()

        f0 = self.freq_finder()

        remf, immf, sremf, simmf, oput = self.optimization(time,flux,f0)

        plt.figure()
        plt.title('lombscargle periodogram of {}'.format(self.id))
        self.lombscargle_periodogram(time,simmf)
        # plt.xlim(2,4)
        # plt.ylim(1e-8, 1e-1)
        plt.legend()
        plt.savefig('./stellar_radio_plot_{}_{}.pdf'.format(self.instrument,str(self.id)))
