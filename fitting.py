import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import glob
from scipy.optimize import curve_fit

def LOGSAT(log_Mh, log_Mcut, log_M1, alpha):    
    i = 0
    while log_Mh[i] < log_Mcut:
        i += 1
    arr = alpha * (np.log10(10.**log_Mh[i:] - 10.**log_Mcut) - log_M1)
    if i != 0: return np.concatenate(([-1.6 for j in range(i) ], arr))
    else: return arr

def LOGCEN(log_Mh, log_Mmin, sigma):
    erf = sc.special.erf((log_Mh - log_Mmin)/sigma)
    if np.any(erf < -1):
        print erf
    return np.log10(0.5 * (1 + erf))

def Adjust (X_axis, Cen, Sat):

    log_Mh = X_axis
    mask_cen = Cen > -1
    mask_sat = Sat > -1

    x0_mcut = log_Mh[mask_sat][0] - 0.1
    x0_cen = np.array([11.2, 0.6])
    x0_sat = np.array([x0_mcut, 13.2, 1.3])

    popt_cen, pcov_cen = curve_fit(LOGCEN, log_Mh[mask_cen], Cen[mask_cen], x0_cen)
    popt_sat, pcov_sat = curve_fit(LOGSAT, log_Mh[mask_sat], Sat[mask_sat], x0_sat)

    perr_cen = np.sqrt(np.diag(pcov_cen))
    perr_sat = np.sqrt(np.diag(pcov_sat))
    
    par_cen = np.array([popt_cen, perr_cen]).T
    par_sat = np.array([popt_sat, perr_sat]).T
    
    return par_cen, par_sat
    