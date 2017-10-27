import numpy as np


def psd2timeSerie(psdall, fixedPhase=None):
    '''
    Generate time serie

    # if fixedPhase is None, then the phase is random
    '''
    # convert PSD to time serie
    # psdtot = np.array([psdall, psdall[2::-1]])[::-1]
    # psdtot = np.concatenate((psdall[::-1], psdall[1:]))
    psdtot = psdall
    # norm = np.sum(psdtot)

    amp = np.sqrt(psdtot * 2)   # %2 because the psd has been doubled by adding negative values
    # random phase between [0, 2pi]
    #  if fixedPhase is None:
    phase = np.random.random(len(amp)) * 2 * np.pi
    #     else:
    #         phase = np.ones(len(amp)) * fixedPhase

    h = amp * np.exp(1j * phase)

    ts = np.real(np.fft.fft(h))

    return ts


def generatePSD(samplingTime, length,
                white_rms, f_rms,
                peak, peak_rms, peak_width=None):
    '''
    based on the YAO implementation

    Three types of vibrations are implemented:
        - white noise rms vibration
        - 1/f rms vibration
        - peak in the PSD (peak location, peak rms, peak width)

    Parameter
    ---------
    samplingTime : float
        loop sampling time [sec]
    length : int
        number of point desired in generated time serie

    white_rms : float
        rms of white noise [arcsec]
    f_rms: float
        rms of 1/f noise (from 1Hz included up to cutoff frequency)
    peak : float, ndarray
        vector containing the frequency [Hz] of the peaks
    peak_rms : float, ndarray
        vector of rms of each peak [arcsec]
    peak_width: float, ndarray
        vector of width [fwhm in Hz] for each peak. Default is 1Hz

    Returns
    -------
    float, ndarray
        time serie
    '''

    # 1. Generate the PSD
    nbOfPt = int(length / 2 + 1)
    freq = np.linspace(0, 1 / samplingTime / 2, nbOfPt)
    psdall = np.zeros(nbOfPt)

    # white noise
    psd = np.ones(nbOfPt)
    psd = psd / np.sum(psd) * white_rms**2
    psdall = np.copy(psd)

    # 1/f noise  (nb: TBC - simplified wrt yao)
    psd = 1 / freq[1:]
    psd = psd / np.sum(psd) * f_rms**2
    psdall[1:] += psd

    # peaks
    assert len(peak) == len(peak_rms)
    if peak_width is None:
        peak_width = np.ones(len(peak))

    for i in range(0, len(peak)):
        if (peak[i] < 0) or (peak[i] > np.max(freq)):
            pass
        else:
            peak_width[i] = np.max([peak_width[i], freq[1] / 10])
            sigma = peak_width[i] / 2.35
            psd = np.exp(-(freq - peak[i])**2 / (2 * sigma**2))
            psd = psd / np.sum(psd) * peak_rms[i]**2
            psdall += psd
    return freq, psdall


def generate_vibration_time_series(samplingTime, length,
                                   white_rms, f_rms,
                                   peak, peak_rms, peak_width=None):
    '''
    based on the YAO implementation

    Three types of vibrations are implemented:
        - white noise rms vibration
        - 1/f rms vibration
        - peak in the PSD (peak location, peak rms, peak width)

    sampling_time : integration time
    legnth: number of loop iterations.... (if infinite simu problem...)

    Parameter
    ---------
    samplingTime : float
        loop sampling time [sec]
    length : int
        number of point desired in generated time serie

    white_rms : float
        rms of white noise [arcsec]
    f_rms: float
        rms of 1/f noise (from 1Hz included up to cutoff frequency)
    peak : float, ndarray
        vector containing the frequency [Hz] of the peaks
    peak_rms : float, ndarray
        vector of rms of each peak [arcsec]
    peak_width: float, ndarray
        vector of width [fwhm in Hz] for each peak. Default is 1Hz

    Returns
    -------
    float, ndarray
        time serie
    '''

    # 1. Generate the PSD
    freq, psdall = generatePSD(samplingTime, length, white_rms, f_rms,
                               peak, peak_rms, peak_width)

    # 2. convert psd to time serie
    ts = psd2timeSerie(psdall)

    return ts


def generate2Ddisturbance(samplingTime, length, peak, peak_rms, phase, theta):
    '''
    phase : temporal phase
    theta : vibration axis (defined by the position angle)
    '''
    time = np.arange(0, length * samplingTime, samplingTime)
    tipall = np.zeros(len(time))
    tiltall = np.zeros(len(time))
    for i in range(len(peak)):
        Ax = peak_rms[i] * np.cos(theta[i])
        Ay = peak_rms[i] * np.sin(theta[i])
        tip = Ax * np.cos(2 * np.pi * peak[i] * time + phase[i])
        tilt = Ay * np.cos(2 * np.pi * peak[i] * time + phase[i])

        tipall += tip
        tiltall += tilt
    return tipall, tiltall
