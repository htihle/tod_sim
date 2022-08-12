import numpy as np
import numpy.fft as fft

def generate_1f_noise(
        n_samples,
        PSD_params,
        sampleRate=50):
    sigma_0, f_knee, alpha = PSD_params
    n_tot = int(1.5 * n_samples)
    f = fft.rfftfreq(n_tot) * sampleRate
    data = np.random.randn(len(f)) * np.sqrt(n_tot)
   
    SN = (f / f_knee) ** (alpha)
    PowerSpectrum = sigma_0 ** 2 * (1.0 + SN) 
    PowerSpectrum[0] = 0
    A = np.sqrt(PowerSpectrum)
    data = fft.irfft(data * A)

    return data[:n_samples] - np.mean(data[:n_samples])