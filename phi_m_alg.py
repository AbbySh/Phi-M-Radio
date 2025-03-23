import numpy as np
from astropy.timeseries import LombScargle

class phi_m_radio():

    def tune(ts,ys,qmin=0.,qmax=np.Inf):
        """
        Finds frequency based off of LombScargle peaks.
        Fits parabola, refines peak that corresponds to orbital period of companion.

        Args:
            time (np.ndarray): Time data.
            flux (np.ndarray): Flux data.
            qmin (float): Minimum for frequency choice.
            qmax (float): Maximum for frequency choice.

        Returns:
            float: Refined x-value of peak, ie refined orbital period.
            
        Bugs:
            Requires evenly spaced input data (evenly in x)
        """            
    
        q, y = LombScargle(ts, ys).autopower()
        pk_idx = np.argmax(y * (q > qmin) * (q < qmax))
        pk_x = q[pk_idx]

        y_three = y[pk_idx - 1 : pk_idx + 2] # this dies if pk_idx is at the array edge
        delta = 0.5 * (q[pk_idx + 1] - q[pk_idx - 1])
        assert np.shape(y_three) == (3,)
        b = (y_three[2] - y_three[0]) / (2. * delta)
        a = (y_three[0] - 2.0 * y_three[1] + y_three[2]) / (delta ** 2)
        assert a<0
        assert (-delta) < (-b / a) < delta
        refined_x = pk_x - b / a

        return refined_x

    def demodulate(ts,ys,mode_f):
        """
        Performs phase demodulation (flux * e^{2pi i f t} calculation).

        Arguments:
            ts (np.array): Array of times
            ys (np.array): Array of star's flux
            mode_f (float): Pulsation mode 
        
        Returns:
            np.array: Array of complex numbers, flux multiplied by e^{2pi i f t}
        """
        qs = (ys-np.nanmean(ys)) * np.exp(2.*np.pi*1j*mode_f*ts)
        return qs/np.nanmean(qs)

    def listen(ts,qs,mode_f):
        """
        Gets lombscargle periodogram data from qs calculated in the 'radio' function.

        Arguments:
            ts (np.array): Array of times
            qs (np.array): Result from 'radio' function, flux * e^{2 pi i f t} / mean(flux * e^{2 pi i f t})
            mode_f (float): Pulsation mode 
        
        Returns:
            np.array: Array of frequency range (1/observed time period in days)
            np.array: Array of lombscargle output 
        """
        delta_f = 1. / (max(ts) - min(ts))
        fs = np.arange(delta_f, mode_f, delta_f / 3.0)
        ps = LombScargle(ts,qs).power(fs,normalization='psd')
        return fs, ps
