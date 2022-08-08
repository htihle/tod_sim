import numpy as np

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
            delta_nu=2.0/1024  # GHz
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

    def generate_tod(self, t_comps=None):
        if t_comps is None:
            comps = self.temp_components
        else:
            comps = self.temp_components[t_comps]
        temp_tod = np.zeros((self.n_feed, self.n_sb, self.n_freq, self.n_tod))
        for comp in comps.values():
            temp_tod += comp.generate_tod()
        self.tod = self.gain.generate_gain() * temp_tod \
            * (1.0 + self.radiometer_noise * np.random.randn(
                self.n_feed, self.n_sb, self.n_freq, self.n_tod
            ))
        return self.tod

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
            else:
                print('Unknown gain name', gain)

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