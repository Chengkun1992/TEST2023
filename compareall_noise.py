import numpy as np
import matplotlib.pyplot as plt


# from IPython.core.pylabtools import figsize
plt.rcParams['figure.dpi'] = 100

data = np.loadtxt('compareall_noise.csv', delimiter='\t', skiprows=1)

data=data.transpose()
print(data.shape)

print(data)

f=plt.figure()
f.set_size_inches(9, 12)


legfont = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 11,
}


labfont = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

tickfont= {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

titlefont= {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}


name_list = ['MMVAs','WC2014','MMVAs-N','WC2014-N']


x =list(range(len(name_list)))
total_width, n = 0.7, 6
width = total_width / n

# grid = plt.GridSpec(1,  hspace=1)
# ax=plt.subplot(grid[0, :7])

# grid = plt.GridSpec(15, 1, wspace=1, hspace=3.9)

grid = plt.GridSpec(8, 1, hspace=-0.2)

ax=plt.subplot(grid[:2, 0])
# ax=plt.subplot(grid[0, :7])

# plt.subplot(311)
plt.bar(x, data[0], width=width, label='HashIP-G', fc = 'w', hatch='xxxxx', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[3], width=width, label='HashIP-S', fc = 'k', hatch='-', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[6], width=width, label='HashIP-R',  fc = 'w', hatch='......', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[9], width=width, label='CNN', fc = 'w', tick_label = name_list, hatch='/////', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[12], width=width, label='EventPre', fc = 'w', hatch='+++++', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[15], width=width, label='TSBP', fc = 'w', hatch='oo', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[18], width=width, label='HashPre', fc = 'w', hatch='***', ec='k', ls='-', lw=1)

plt.ylim(0,0.7)
plt.xticks(family='Times New Roman')
plt.yticks(family='Times New Roman')
plt.title('(a) RMSE',titlefont)
plt.tick_params(labelsize=12,pad=1)
plt.legend(prop=legfont,ncol=2,\
           handlelength=0.7,\
           labelspacing=0.2,\
           handletextpad=0.2,\
           columnspacing=0.2,\
           borderpad=0.15,\
           loc=2)



ax=plt.subplot(grid[3:5, 0])

plt.bar(x, data[1], width=width, label='HashIP-G', fc = 'w', hatch='xxxxx', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[4], width=width, label='HashIP-S', fc = 'k', hatch='-', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[7], width=width, label='HashIP-R',fc = 'w', hatch='......', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[10], width=width, label='CNN',fc = 'w', tick_label = name_list, hatch='/////', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[13], width=width, label='EventPre', fc = 'w', hatch='+++++', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[16], width=width, label='TSBP', fc = 'w', hatch='oo', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[19], width=width, label='HashPre', fc = 'w', hatch='***', ec='k', ls='-', lw=1)



plt.ylim(0,0.6)
plt.xticks(family='Times New Roman')
plt.yticks(family='Times New Roman')
plt.title('(b) MAE',titlefont)
plt.tick_params(labelsize=12,pad=1)
plt.legend(prop=legfont,ncol=2,\
           handlelength=0.7,\
           labelspacing=0.2,\
           handletextpad=0.2,\
           columnspacing=0.2,\
           borderpad=0.15,\
           loc=2)
#frameon=False


ax=plt.subplot(grid[6:, 0])
plt.bar(x, data[2], width=width, label='HashIP-G', fc = 'w', hatch='xxxxx', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[5], width=width, label='HashIP-S',fc = 'k', hatch='-', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[8], width=width, label='HashIP-R',fc = 'w', hatch='......', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[11], width=width, label='CNN', tick_label = name_list, fc = 'w', hatch='/////', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[14], width=width, label='EventPre',  fc = 'w', hatch='+++++', ec='k', ls='-', lw=1)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[17], width=width, label='TSBP', fc = 'w', hatch='oo', ec='k', ls='-', lw=1)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, data[20], width=width, label='HashPre', fc = 'w', hatch='***', ec='k', ls='-', lw=1)

plt.ylim(0,7.5)
plt.xticks(family='Times New Roman')
plt.yticks(family='Times New Roman')
plt.title('(c) MAPE',titlefont)
plt.tick_params(labelsize=12,pad=1)
plt.legend(prop=legfont,ncol=2,\
           handlelength=0.7,\
           labelspacing=0.2,\
           handletextpad=0.2,\
           columnspacing=0.2,\
           borderpad=0.15,\
           loc=2)
plt.subplots_adjust(wspace =0.15, hspace =0)

# f.savefig("Figures/compareall_noise.pdf", bbox_inches='tight', pad_inches=0 )

plt.show()

