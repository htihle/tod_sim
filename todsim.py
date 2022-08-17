import numpy as np
import h5py

import tools

class TodSim():
    def __init__(
            self,
            n_tod,
            n_freq=1024,
            n_sb=4,
            n_feed=19,
            gain=None,
            temp_components=None,
            pointing=None,
            sample_rate=50,  # Hz
            delta_nu=2.0/1024,  # GHz
            freqs_linear=False,
            t_hot=300.0, 
        ):
        if gain is not None:
            self.gain = gain
        if temp_components is not None:
            self.temp_components = temp_components
        else:
            self.temp_components = {}
        self.n_tod = n_tod
        self.n_freq = n_freq
        self.n_sb = n_sb
        self.n_feed = n_feed
        self.radiometer_noise = 1.0 / np.sqrt(delta_nu / sample_rate * 1e9)
        self.pointing = pointing
        self.freqs_linear = freqs_linear
        self.t_cmb = 2.7255  # K
        self.t_hot = t_hot

    def generate_tod(self, t_comps=None):
        if t_comps is None:
            comps = self.temp_components
        else:
            comps = self.temp_components[t_comps]
        temp_tod = np.zeros((self.n_feed, self.n_sb, self.n_freq, self.n_tod))
        for comp in comps.values():
            temp_tod += comp.generate_tod()

        # add Tsys spikes here at some point

        self.tod = self.gain.generate_gain() * temp_tod \
            * (1.0 + self.radiometer_noise * np.random.randn(
                self.n_feed, self.n_sb, self.n_freq, self.n_tod
            ))
        self.p_cold = np.mean(self.tod, -1)
        self.t_sys = np.mean(temp_tod, -1)
        self.gain_nu = self.gain.gain_nu
        return self.tod

    def get_calibration_info(self):
        self.p_hot = self.p_cold + self.gain_nu * (self.t_hot - self.t_cmb)
        return self.t_sys, self.t_hot, self.t_cmb, self.p_hot, self.p_cold


    def add_component(self, comp):
        if isinstance(comp, str):
            if comp == 'constant_radiometer':
                self.temp_components[comp] = ConstantRadiometer(self)
            elif comp == 'sinus_wn_standing_wave':
                self.temp_components[comp] = SinusWNStandingWave(self)
            else:
                print('Unknown component name', comp)
        else:
            self.temp_components[comp.name] = comp

    def add_gain(self, gain):
        if isinstance(gain, str):
            if gain == 'constant_gain':
                self.gain = ConstantGain(self)
            elif gain == 'handmade_gain':
                self.gain = HandmadeGain(self)
            else:
                print('Unknown gain name:', gain)

class Gain():
    def __init__(
            self, 
            sim,
        ):
        self.sim = sim

    def generate_gain(self):
        pass


class TempComponent():
    def __init__(
            self, 
            sim,
        ):
        self.sim = sim

    def generate_tod(self):
        pass


class ConstantGain(Gain):
    def __init__(self, sim):
        super().__init__(sim)
        self.name = 'constant_gain'

    def generate_gain(self):
        return np.ones((
            self.sim.n_feed, self.sim.n_sb,
            self.sim.n_freq, self.sim.n_tod
            ))



class HandmadeGain(Gain):
    def __init__(self, sim, gain_PSD_params=None):
        super().__init__(sim)
        self.name = 'handmade_gain'
        self.gain_PSD_params = gain_PSD_params

    @staticmethod
    def _generate_random_gain_db(n_freq):
        nums = np.random.randn(10)

        nu = np.linspace(0, 4, 2 * n_freq)
        gain_nu_db = 40 - 4 * (nu - 2.0) ** 2 \
            + (nums[0] + nums[1] * (nu - 2.0) + nums[2] * (nu - 2.0) ** 2) \
            * np.sin(2 * np.pi * nu / 0.6 + 3 * nums[3]) ** 5 \
            + nums[4] * np.sin(2 * np.pi * nu / 2e-1) ** 3
        sb_offset = np.zeros(2 * n_freq) + 0.5
        sb_offset[:n_freq] = -0.5
        gain_nu_db = gain_nu_db + nums[8] * sb_offset
        gain_nu_db = gain_nu_db * min(max(1 + 0.15 * nums[9], 0.5), 1.5)
        return gain_nu_db
    
    def generate_gain(self):
        assert(self.sim.n_sb == 4, 'Gain model assumes four sidebands')
        self.gain_nu = np.zeros((self.sim.n_feed, self.sim.n_sb, self.sim.n_freq))
        self.gain_t = np.zeros((self.sim.n_feed, self.sim.n_tod))
        if self.gain_PSD_params is None:
            filename = 'Cf_prior_data.hdf5'
            with h5py.File(filename, mode="r") as my_file:
                alpha_prior = np.array(my_file['alpha_prior'][()])
                fknee_prior = np.array(my_file['fknee_prior'][()])
                sigma0_prior = np.array(my_file['sigma0_prior'][()])
            self.sim.gain_psd_params = np.array([sigma0_prior, fknee_prior, alpha_prior]).transpose((1, 0))

        for feed in range(self.sim.n_feed):
            self.gain_t[feed] = tools.generate_1f_noise(self.sim.n_tod, self.sim.gain_psd_params[feed])
            for band in range(2):
                gain_nu_db = self._generate_random_gain_db(self.sim.n_freq)
                gain_nu_band = 10.0 ** (gain_nu_db / 10.0)
                if not self.sim.freqs_linear:
                    gain_nu_band[:self.sim.n_freq] = gain_nu_band[:self.sim.n_freq][::-1]
                self.gain_nu[feed, 2*band] = gain_nu_band[:self.sim.n_freq]
                self.gain_nu[feed, 2*band+1] = gain_nu_band[self.sim.n_freq:]
        return self.gain_nu[:, :, :, None] * (1.0 + self.gain_t[:, None, None, :])


class ConstantRadiometer(TempComponent):
    def __init__(self, sim, temp_receiver=20.0):
        super().__init__(sim)
        self.name = 'constant_radiometer'
        self.temp_receiver = temp_receiver

    def generate_tod(self):
        return self.temp_receiver * np.ones((
            self.sim.n_feed, self.sim.n_sb,
            self.sim.n_freq, self.sim.n_tod
            ))

class SinusWNStandingWave(TempComponent):
    def __init__(self, sim, period=0.4, amplitude=0.1):
        super().__init__(sim)
        self.name = 'sinus_wn_standing_wave'
        self.period = period
        self.amplitude = amplitude

    def generate_tod(self):
        tod = np.random.randn(self.sim.n_tod) * self.amplitude
        offsets = np.random.randn(self.sim.n_feed, self.sim.n_sb) * 2 * np.pi
        freqs = np.linspace(0, 2, self.sim.n_freq + 1)
        freqs = 0.5 * (freqs[1:] + freqs[:-1])
        freq_ampls = np.sin(2 * np.pi / self.period * freqs[None, None, :] + offsets[:, :, None])
        return freq_ampls[:, :, :, None] * tod[None, None, None, :]

