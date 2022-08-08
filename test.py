import numpy as np
import h5py
import matplotlib.pyplot as plt

import todsim

n_tod = 1000

my_sim = todsim.TodSim(n_tod=n_tod)
my_sim.add_gain('constant_gain')
my_sim.add_component('constant_radiometer')
my_sim.add_component('sinus_wn_standing_wave')

full_tod = my_sim.generate_tod()

print(full_tod.shape)

plt.imshow(full_tod[0, 0])
plt.show()