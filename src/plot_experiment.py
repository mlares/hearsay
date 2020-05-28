from hearsay import hearsay
import pickle
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


####################################################
# Figuras 3 y 4
####################################################


if len(argv) > 1:
    conf = hearsay.Parser(argv[1])
else:
    conf = hearsay.Parser()

G = hearsay.C3Net(conf)
G.set_parameters()
R = hearsay.Results(G)
R.load()  


R.redux_2d()

fn = R.config.filenames
fname = fn.dir_output + fn.exp_id
fname1 = fname + '/m1.pk'
fname2 = fname + '/m2.pk'
 
with open(fname1, 'rb') as pickle_file:
   m1 = pickle.load(pickle_file)
with open(fname2, 'rb') as pickle_file:
   m2 = pickle.load(pickle_file)



# PLOT M1 ***************************************************************

mt = np.transpose(m1)
sigma=[2,2]
mt_smoothed=ndimage.filters.gaussian_filter(mt, sigma)

A = R.params['tau_awakening']
S = R.params['tau_survive']

N1 = len(R.params['tau_awakening'])
Amin = min(R.params['tau_awakening'])
Amax = max(R.params['tau_awakening'])
dA = (Amax-Amin)/N1
Amin = Amin - dA/2
Amax = Amax + dA/2

N2 = len(R.params['tau_survive'])
Smin = min(R.params['tau_survive'])
Smax = max(R.params['tau_survive'])
dS = (Smax-Smin)/N2
Smin = Smin - dS/2
Smax = Smax + dS/2

levels = list(np.arange(np.min(m1), np.max(m1), (np.max(m1)-np.min(m1))/20 ))

mmin = m1.min()
mmax = m1.max()

plt.close('all')
fig, ax = plt.subplots()

#im = ax.imshow(m1, origin='lower', aspect='auto',
im = ax.imshow(mt_smoothed, origin='lower', aspect='auto',
        interpolation='kaiser',
        extent=[Amin,Amax,Smin,Smax],
        vmin=mmin, vmax=mmax,
        cmap=cm.viridis)

CS = ax.contour(mt_smoothed, levels=levels, colors='green',
        extent=[Amin,Amax,Smin,Smax],
        linewidths=1.5, alpha=.3)

ax.clabel(CS, list(CS.levels[5:]), inline=1, fontsize=10, fmt='%1.1f')
#plt.xticks(A[::7], [str(int(a)) for a in A[::7]])
#plt.yticks(S[::7], [str(int(a)) for a in S[::7]])

# plt.title(titulo)
plt.xlabel('awakening rate [kyr]')
plt.ylabel('survival rate [kyr]')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()
 
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 20
plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
fig.savefig('../plt/plot_M1.png', format='png')




# PLOT M2 **********************************************************

mt = np.transpose(m2)
sigma=[2,2]
mt_smoothed=ndimage.filters.gaussian_filter(mt, sigma)

A = R.params['tau_awakening']
S = R.params['tau_survive']

N1 = len(R.params['tau_awakening'])
Amin = min(R.params['tau_awakening'])
Amax = max(R.params['tau_awakening'])
dA = (Amax-Amin)/N1
Amin = Amin - dA/2
Amax = Amax + dA/2

N2 = len(R.params['tau_survive'])
Smin = min(R.params['tau_survive'])
Smax = max(R.params['tau_survive'])
dS = (Smax-Smin)/N2
Smin = Smin - dS/2
Smax = Smax + dS/2

levels = list(np.arange(np.min(m2), np.max(m2), (np.max(m2)-np.min(m2))/20 ))

mmin = m2.min()
mmax = m2.max()

plt.close('all')
fig, ax = plt.subplots()

#im = ax.imshow(m2, origin='lower', aspect='auto',
im = ax.imshow(mt_smoothed, origin='lower', aspect='auto',
        #interpolation='kaiser',
        extent=[Amin,Amax,Smin,Smax],
        vmin=mmin, vmax=mmax,
        cmap=cm.viridis)

CS = ax.contour(mt_smoothed, levels=levels, colors='green',
        extent=[Amin,Amax,Smin,Smax],
        linewidths=1.5, alpha=.3)

ax.clabel(CS, list(CS.levels[5:]), inline=1, fontsize=10, fmt='%1.1f')
plt.xlabel('awakening rate [kyr]')
plt.ylabel('survival rate [kyr]')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 20
plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
fig.savefig('../plt/plot_M2.png', format='png') 

