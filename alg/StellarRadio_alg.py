import numpy as np
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
import scipy.ndimage
import pickle
from itertools import combinations

class StellarRadioAlg:
    """
    Explanation of what this code does.
    """

    def __init__(self,instrument,id,amp0=1000,phase0=-1,iters=4): #MAGIC NUMBERS
        self.instrument = instrument
        self.id = id
        self.amp0 = amp0
        self.phase0 = phase0
        self.iters = iters

        if self.instrument == 'kepler':
            self.ins_prefix = 'kic'
        elif self.instrument == 'tess':
            self.ins_prefix = 'tic'

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
        # print('mix:',self.amp0,f0,self.phase0,np.median(flux),np.min(flux),np.mean(flux),np.mean(mf))
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

            #refine amplitude guess
            new_amp = self.amp0 / np.sqrt(np.mean(sremf**2 + simmf**2))
            # print(np.mean(sremf**2 + simmf**2),np.mean(sremf**2),np.mean(simmf**2))
            # print("Amps:", self.amp0, new_amp)
            self.amp0 = new_amp
            
            #refine phase guess
            sremf,simmf = self.mix(time,flux,f0)
            new_phase = self.phase0 + np.arctan2(np.mean(simmf), np.mean(sremf))
            # print("Phases:", self.phase0, new_phase)
            self.phase0 = new_phase
        
            #refine sremf,simmf
            sremf, simmf = self.mix(time,flux,f0)

            output=minimize(self.objective,f0,args=(time,flux,self.amp0,self.phase0),method="Nelder-Mead")
            # print("Frequencies:", f0, np.mean((output.final_simplex[0][0],output.final_simplex[0][1])))
            # f0=np.mean((output.final_simplex[0][0],output.final_simplex[0][1]))
            
        return simmf

    def ls_period(self,time,simmf):
        """
        Performs LombScargle periodogram.

        Args:
            time(np.ndarray): Time data.
            simmf(np.ndarray):

        Returns:
            np.ndarray:
            np.ndarray:
        """
        q,y = LombScargle(time,simmf).autopower()
        return q,y

    # def period_corrob(self,modes,orbital_periods):
    #     """
    #     Checks if two modes produce similar orbital periods in resulting LombScargle.

    #     Args:
    #         modes(np.ndarray):
    #         orbital_periods(np.ndarray):

    #     Returns:
    #         float: 
    #         float:
    #     """
    #     #1. peak finder (or fit gaussian) on all output lombscargles per kic
    #     #2. comparison of peaks of all lombscargles, if within certain bounds set off YAY!
    #     combos = list(combinations(orbital_periods, 2))
    #     for no,i in enumerate(combos):
    #         diff = abs(combos[no][0]-combos[no][1])
    #         if diff <= ____: #SOME VALUE
    #             per1,per2 = i[0],i[1]
    #             per1idx,per2idx = orbital_periods.index[per1],orbital_periods.index[per2]
    #             mode1,mode2 = modes[per1idx],modes[per2idx]
    #             print('{} and {} show similar orbital periods!'.format(mode1,mode2))
    #             = ([mode1,mode2])

    def run_all_steps(self,time,flux,modes,mode_type='exact'): 
        """Runs algorithm steps in order, which ends in plotting a LombScargle
        periodogram of time vs simmf.

        Args:
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            f0 (float): Exact frequency (mode) value. 

        Returns:
            np.ndarray: Time data.
            np.ndarray: Flux data.
            np.ndarray: Number of light curve sectors.
            np.ndarray: Optimized Gaussian-filtered immf.
        """
        # if mode_type == 'rough':
        #     y_array = 
        #     x1 = 
        #     delta = 
        #     x_adjust = ru.refine_peak()
        q_modes,y_modes = LombScargle(time,flux).autopower()

        if mode_type == 'exact':
            simmfs = []
            qs = []
            ys = []

            for f in modes:
                print('Calculating for f = {}'.format(f))
                simmf = self.optimization(time,flux,f)

                q,y = self.ls_period(time,simmf)
                qs.append(q)
                ys.append(y)
                # orb_per = 1./q

                # orbital_periods.append(orb_per)

        ## 1. do comparison of peaks of all periods (need two to match at least)
        # 1.5. plot all LS periodograms per star over each other - do in plotting script
        ## 2. check if the phase variations match/don't match

        plot_vals = {'flux':flux,'time':time,'modes':modes,'simmfs':simmfs,'qs':qs,'ys':ys,'q_modes':q_modes,'y_modes':y_modes}
        with open('./pickle_files/pickle_{}_{}.pkl'.format(self.ins_prefix,self.id),'wb') as file:
            pickle.dump(plot_vals,file)
            file.close()
        print ('done')
        return qs,ys,q_modes,y_modes
