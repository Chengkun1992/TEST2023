import numpy as np
import matplotlib.pyplot as plt

# from IPython.core.pylabtools import figsize
plt.rcParams['figure.dpi'] = 100

data = np.loadtxt('eff_datasize_noise.csv', delimiter='\t', skiprows=1)
data = data.transpose()
print(data)

x = data[0]
print(x)

f = plt.figure()
f.set_size_inches(10, 7)

legfont = {'family': 'Times New Roman',
           'weight': 'normal',
           'size': 11,
           }

labfont = {'family': 'Times New Roman',
           'weight': 'normal',
           'size': 14,
           }

tickfont = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }

titlefont = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14,
             }

grid = plt.GridSpec(2, 17, wspace=10.1, hspace=0.35)
ax = plt.subplot(grid[0, :8])

# ax=plt.subplot(141)
HashPre, = plt.plot(x, data[1], '-v', label='HashPre', color='k')
EventPre, = plt.plot(x, data[2], '--*', label='EventPre', color='k')
Tsbp, = plt.plot(x, data[3], '--*', label='EventPre', color='k')
CNN, = plt.plot(x, data[4], '--*', label='EventPre', color='k')
HashIP, = plt.plot(x, data[5], ':o', label='HashIP', color='k')
HashIPdyn, = plt.plot(x, data[6], '-.s', label='HashIP-dyn', color='k')
plt.xlabel('number of time slots', labfont)
plt.title('(a) MMVAs', titlefont)
plt.xticks(x, rotation='0')
plt.xticks(family='Times New Roman')
plt.yticks(family='Times New Roman')
ax.set_ylabel('time (ms)', fontsize=12, labelpad=0, family='Times New Roman')

# ymajorLocator = MultipleLocator(10)
# ax.yaxis.set_major_locator(ymajorLocator)
# yminorLocator = MultipleLocator(5)
# ax.yaxis.set_minor_locator(yminorLocator)

plt.xlim((1, 30))
plt.ylim((5, 40))
plt.legend(handles=[HashPre, EventPre, Tsbp, CNN, HashIP, HashIPdyn], prop=legfont, \
           ncol=2, \
           handlelength=1.5, \
           labelspacing=0.2, \
           handletextpad=0.2, \
           columnspacing=0.2, \
           borderpad=0.15, \
           loc=3)
plt.tick_params(labelsize=12, pad=1)

ax = plt.subplot(grid[0, 9:])
# ax=plt.subplot(142)
ax.set_ylabel('time (ms)', fontsize=12, labelpad=0, family='Times New Roman')
HashPre, = plt.plot(x, data[5], '-v', label='HashPre', color='k')
EventPre, = plt.plot(x, data[6], '--*', label='EventPre', color='k')
Tsbp, = plt.plot(x, data[3], '--*', label='EventPre', color='k')
CNN, = plt.plot(x, data[4], '--*', label='EventPre', color='k')
HashIP, = plt.plot(x, data[7], ':o', label='HashIP', color='k')
HashIPdyn, = plt.plot(x, data[8], '-.s', label='HashIP-dyn', color='k')
plt.xlabel('number of time slots', labfont)
# plt.ylabel('MS',labfont)
plt.xticks(x, rotation='0')
plt.xticks(family='Times New Roman')
plt.yticks(family='Times New Roman')

# ymajorLocator = MultipleLocator(10)
# ax.yaxis.set_major_locator(ymajorLocator)
# yminorLocator = MultipleLocator(5)
# ax.yaxis.set_minor_locator(yminorLocator)

plt.title('(b) WC2014', titlefont)
plt.xlim((1, 30))
plt.ylim((10, 60))
plt.legend(handles=[HashPre, EventPre, Tsbp, CNN, HashIP, HashIPdyn], prop=legfont, \
           handlelength=1.5, \
           ncol=2, \
           labelspacing=0.2, \
           handletextpad=0.2, \
           columnspacing=0.2, \
           borderpad=0.15, \
           loc=2)
plt.tick_params(labelsize=12, pad=1)

ax = plt.subplot(grid[1, :8])
# ax=plt.subplot(143)
ax.set_ylabel('time (ms)', fontsize=12, labelpad=0, family='Times New Roman')
HashPre, = plt.plot(x, data[5], '-v', label='HashPre', color='k')
EventPre, = plt.plot(x, data[6], '--*', label='EventPre', color='k')
Tsbp, = plt.plot(x, data[3], '--*', label='EventPre', color='k')
CNN, = plt.plot(x, data[4], '--*', label='EventPre', color='k')
HashIP, = plt.plot(x, data[7], ':o', label='HashIP', color='k')
HashIPdyn, = plt.plot(x, data[8], '-.s', label='HashIP-dyn', color='k')
plt.xlabel('number of time slots', labfont)
# plt.ylabel('MS',labfont)
plt.xticks(x, rotation='0')
plt.xticks(family='Times New Roman')
plt.yticks(family='Times New Roman')

# ymajorLocator = MultipleLocator(10)
# ax.yaxis.set_major_locator(ymajorLocator)
# yminorLocator = MultipleLocator(5)
# ax.yaxis.set_minor_locator(yminorLocator)

plt.title('(c) MMVAs-N', titlefont)
plt.xlim((1, 30))
plt.ylim((10, 60))
plt.legend(handles=[HashPre, EventPre, Tsbp, CNN, HashIP, HashIPdyn], prop=legfont, \
           handlelength=1.5, \
           ncol=2, \
           labelspacing=0.2, \
           handletextpad=0.2, \
           columnspacing=0.2, \
           borderpad=0.15, \
           loc=2)
plt.tick_params(labelsize=12, pad=1)

ax = plt.subplot(grid[1, 9:])
# ax=plt.subplot(144)
ax.set_ylabel('time (ms)', fontsize=12, labelpad=0, family='Times New Roman')
HashPre, = plt.plot(x, data[5], '-v', label='HashPre', color='k')
EventPre, = plt.plot(x, data[6], '--*', label='EventPre', color='k')
Tsbp, = plt.plot(x, data[3], '--*', label='EventPre', color='k')
CNN, = plt.plot(x, data[4], '--*', label='EventPre', color='k')
HashIP, = plt.plot(x, data[7], ':o', label='HashIP', color='k')
HashIPdyn, = plt.plot(x, data[8], '-.s', label='HashIP-dyn', color='k')
plt.xlabel('number of time slots', labfont)
# plt.ylabel('MS',labfont)
plt.xticks(x, rotation='0')
plt.xticks(family='Times New Roman')
plt.yticks(family='Times New Roman')

# ymajorLocator = MultipleLocator(10)
# ax.yaxis.set_major_locator(ymajorLocator)
# yminorLocator = MultipleLocator(5)
# ax.yaxis.set_minor_locator(yminorLocator)

plt.title('(d) WC2014-N', titlefont)
plt.xlim((1, 30))
plt.ylim((10, 60))
plt.legend(handles=[HashPre, EventPre, Tsbp, CNN, HashIP, HashIPdyn], prop=legfont, \
           handlelength=1.5, \
           ncol=2, \
           labelspacing=0.2, \
           handletextpad=0.2, \
           columnspacing=0.2, \
           borderpad=0.15, \
           loc=2)
plt.tick_params(labelsize=12, pad=1)

# plt.subplots_adjust(wspace =0.5, hspace =0)


# f.savefig("Figures/eff_datasize_noise.pdf", bbox_inches='tight', pad_inches=0 )
plt.show()
