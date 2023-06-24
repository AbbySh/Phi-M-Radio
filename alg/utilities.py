import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import configparser
from astropy.timeseries import LombScargle


def get_lc_data(id,instrument,cadence):
    """Gets light curve data using Lightkurve.

    Returns:
        np.ndarray: Time data.
        np.ndarray: Flux data.
        np.ndarray: Number of light curve sectors.
    """
    if type(id) != str:
        id = str(id)

    if instrument == 'tess':
        try:
            lc_files = lk.search_lightcurve('TIC'+id, cadence='short').download_all()
            print('cadence type: short')
        except:
            lc_files = lk.search_lightcurve('TIC'+id, cadence='long').download_all()
            print('cadence type: long')

    elif instrument == 'kepler':
        lc_files = lk.search_lightcurve('KIC'+id, cadence=cadence).download_all()
        print('cadence type: {}'.format(cadence))

    else:
        raise TypeError('Only Kepler and TESS are supported instruments')
        
    time, flux = np.array([]), np.array([])
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

        ###mask hack###
        t0 = 0
        per = 58
        for i in np.linspace(0,100):
            event_time = t0 + (i*per)
            mask = (time > (event_time - .5)) & (time < (event_time + .5))
            time = time[~mask]
            flux = flux[~mask]

    flux=flux-1.

    return time,flux


def freq_finder(time,flux,find_mode=False, qmin=0.,qmax=np.Inf):
    """Finds frequency based off of LombScargle peaks. Fits parabola.

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
    # f0_guess = q[np.argmax(y * (q > qmin) * (q < qmax))]
    #do per peak:
    modes = []
    #to get ys, get highest value and one after and one preceding?
    # ys = 
    # x1 = 
    # delta = 
    f0_guess = refine_peak()
    print("Mode guess:",f0_guess)

    return f0_guess

def refine_peak(y,x1,delta):
    """
    Refines peak that corresponds to orbital period of companion.

    Args:
        y(np.ndarray): Array of three y-values corresponding to the y before, of, and after peak y.
        x1(float): X-value of y peak value (or  y[1]).
        delta(flloat): X-axis difference between y[0] and y[1], and y[1] and y[2].

    Returns:
        float: Refined x-value of peak, ie refined orbital period.
    """
    assert y.shape == (3,1)

    b = .5*(y[2]-y[0])/delta
    a = (-2*y[1]+y[0]+y[2])/(delta**2)

    assert a<0
    assert -delta<(-b/a)<delta

    return x1-(b/a)