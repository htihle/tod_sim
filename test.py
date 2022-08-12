import numpy as np
import h5py
import matplotlib.pyplot as plt

import todsim
import tools


n_samples = 100000
tod = tools.generate_1f_noise(n_samples, [1.0, 1e-1, -3])
print(tod.std())
plt.plot(tod)
plt.show()

# filename = 'comap-0030102-2022-07-31-181145_s1.hd5'

# feed = 8
# sb = 2

# with h5py.File(filename, mode="r") as my_file:
#     gain_db = np.array(my_file['tsys/m1/gain'][()])
#     p_hot = np.array(my_file['tsys/m1/avg_hot'][()])
#     p_cold = np.array(my_file['tsys/m1/avg_cold'][()])
#     T_hot = np.array(my_file['tsys/m1/temp_hot'][()])
#     T_cold = np.array(my_file['tsys/m1/temp_cold'][()])

# tsys = (T_hot - T_cold) / (p_hot/p_cold - 1)
# gain = 10 ** (gain_db / 10)

# tsys[:, :, 512] = np.nan
# gain[:, :, 512] = np.nan
# p_cold[:, :, 512] = np.nan
# tsys[:, :, 0] = np.nan
# gain[:, :, 0] = np.nan
# p_cold[:, :, 0] = np.nan
# fig, axs = plt.subplots(3)
# fig.suptitle('comap-0030102-2022-07-31, feed 8, sb 2')
# axs[0].plot(tsys[feed-1, sb-1] * gain[feed-1, sb-1], label='tsys * gain')
# axs[0].plot(p_cold[feed-1, sb-1], '--', label='avg_power')
# # axs[0].title
# axs[0].legend()
# axs[1].plot(tsys[feed-1, sb-1], label='tsys')
# axs[1].legend()
# axs[2].plot(gain[feed-1, sb-1], label='gain')
# axs[2].legend()
# plt.savefig('total_power_gain_tsys.pdf', bbox_inches='tight')
# plt.show()

# N = 41
# fastlen = 55
# x = np.random.normal(0, 1, N)
# for i in range(1,N):
#     x[i] += x[i-1]
# y = np.zeros(fastlen)

# diff = fastlen - N

# halfdiff = diff // 2
# lin = np.linspace(0, 1, diff)
# y[:N] = x

# y[N:-halfdiff] = x[::-1][:diff-halfdiff]
# y[-halfdiff:] = x[::-1][-halfdiff:] - x[::-1][-halfdiff] + x[::-1][diff-halfdiff]
# y[N:] += lin * (x[0] - y[-1])


# plt.plot(y)
# plt.plot(x, ls=":", c="r")
# plt.axvline(N, c="k", ls="--")
# plt.show()
# n_freq = 1024

# def generate_random_gain_db(n_freq):
#     nums = np.random.randn(10)

#     nu = np.linspace(0, 4, 2 * n_freq)
#     gain_nu_db = 40 - 4 * (nu - 2.0) ** 2 \
#         + (nums[0] + nums[1] * (nu - 2.0) + nums[2] * (nu - 2.0) ** 2) \
#         * np.sin(2 * np.pi * nu / 0.6 + 3 * nums[3]) ** 5 \
#         + nums[4] * np.sin(2 * np.pi * nu / 2e-1) ** 3
#     sb_offset = np.zeros(2 * n_freq) + 0.5
#     sb_offset[:n_freq] = -0.5
#     gain_nu_db = gain_nu_db + nums[8] * sb_offset
#     gain_nu_db = gain_nu_db * min(max(1 + 0.15 * nums[9], 0.5), 1.5)
#     return gain_nu_db
    

# for i in range(10):
#     gain_nu_db = generate_random_gain_db(n_freq=n_freq)
#     plt.plot(gain_nu_db)
#     plt.ylim(0, 60)
# plt.savefig('gain_models.png', bbox_inches='tight')
# plt.show()

# n_tod = 1000

# my_sim = todsim.TodSim(n_tod=n_tod)
# my_sim.add_gain('constant_gain')
# my_sim.add_component('constant_radiometer')
# my_sim.add_component('sinus_wn_standing_wave')

# full_tod = my_sim.generate_tod()

# print(full_tod.shape)

# plt.imshow(full_tod[0, 0])
# plt.show()