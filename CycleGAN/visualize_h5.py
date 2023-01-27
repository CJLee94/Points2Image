import h5py
import matplotlib.pyplot as plt

path = './results/cyclegan_gen_v3_color/test_latest/MoNuSeg_test_v3_cyclegan.h5'
hf = h5py.File(path, 'r')

fig = plt.figure()
nfig = len(hf.keys())
for i, key in enumerate(hf.keys()):
    plt.subplot(2, nfig//2, i+1)
    plt.imshow((hf[key][0] + 1.)/2.)
    plt.title(key)

plt.savefig('./results/cyclegan_gen_v3_color/test_latest/images/h5_sanity_check.png')