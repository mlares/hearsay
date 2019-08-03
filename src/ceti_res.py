# coding: utf-8

import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas
import seaborn as sns

from ceti_exp import redux

#===============================================================================
   


#)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.csv')

l1 = (D['tau_awakening']<37000) & (D['tau_survive']<60000) & (D['D_max']==50000.)
l2 = (D['tau_awakening']>85000) & (D['tau_survive']>450000) & (D['D_max']==50000.)
l3 = (D['tau_awakening']<37000) & (D['tau_survive']>450000) & (D['D_max']==50000.)
l4 = (D['tau_awakening']>85000) & (D['tau_survive']<60000) & (D['D_max']==50000.)
#)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))






#lens = []
#for i in list(redux(D)): 
#    lens.append(len(i))
 

# size = the number of cetis
#awaken # lifetime of a CETI
#inbox  # number of contacts in the life of a CETI
#index  # ID identifier

# size = the number of cetis that make contact
#firstc  # time elapsed between awakening and first contact

# size = range in the number of Ncetis (12, fixed)
#count # (mean?) number of contacts per ceti

# size = Number of contacts
# distancias
# hangon
# waiting

# Number of experiments
# ncetis



# D['tau_survive'].unique()
# array([10000., 20000., 30000., 40000., 50000.])
# D['tau_awakening'].unique()
# array([ 4800.,  9600., 14400., 19200., 24000.])
# D['D_max'].unique()
# array([ 1000.,  5000., 10000., 20000., 40000.])
  
#--------------------------------------------------------



# Short range signal samples:
l1s = (D['tau_awakening']<15000) & (D['tau_survive']<15000) & (D['D_max']<=5000.)
l2s = (D['tau_awakening']>20000) & (D['tau_survive']>50000) & (D['D_max']<=5000.)
l3s = (D['tau_awakening']<5000) & (D['tau_survive']>15000) & (D['D_max']<=5000.)
l4s = (D['tau_awakening']>15000) & (D['tau_survive']<26000) & (D['D_max']<=5000.)
# 800, 1050, 1350, 1200


#dense awakening and short lifetime
#sparse awakening and long lifetime
#dense awakening and long lifetime
#sparse awakening and short lifetime

# Long range signal samples:
l1l = (D['tau_awakening']<15000) & (D['tau_survive']<15000) & (D['D_max']>=20000.)
l2l = (D['tau_awakening']>20000) & (D['tau_survive']>50000) & (D['D_max']>=20000.)
l3l = (D['tau_awakening']<5000) & (D['tau_survive']>15000) & (D['D_max']>=20000.)
l4l = (D['tau_awakening']>15000) & (D['tau_survive']<26000) & (D['D_max']>=20000.)
# 1600, 2100, 2721, 2400


##############################################################################

# Distribucion del numero de contactos

bins = np.arange(-1,12)+0.5
A = D['tau_awakening'].unique()
S = D['tau_survive'].unique()
                                                      

d = D[l1]
awaken,inbox1s,distancias,hangon,waiting,count, index, firstc, ncetis, x, y = redux(d) 
d = D[l2]
awaken,inbox2s,distancias,hangon,waiting,count, index, firstc, ncetis, x, y = redux(d) 
d = D[l3]
awaken,inbox3s,distancias,hangon,waiting,count, index, firstc, ncetis, x, y = redux(d) 
d = D[l4]
awaken,inbox4s,distancias,hangon,waiting,count, index, firstc, ncetis, x, y = redux(d) 


plt.hist(inbox1s, bins=bins, histtype='step', align='mid', color='teal',
        label='l1s', linewidth=2)
plt.hist(inbox2s, bins=bins, histtype='step', align='mid',color='slateblue',
        label='l2s', linewidth=2)
plt.hist(inbox3s, bins=bins, histtype='step', align='mid',color='slateblue',
        label='l3s', linewidth=2)
plt.hist(inbox4s, bins=bins, histtype='step', align='mid',color='slateblue',
        label='l4s', linewidth=2)

plt.xticks(np.arange(0, 12, 2))
plt.legend()
plt.show()

#################################################### DIFERENCIAL

d = D[l1]
awaken,inbox1,distancias,hangon,waiting,count,index,firstc1,ncetis,x,y = redux(d)
d = D[l2]
awaken,inbox2,distancias,hangon,waiting,count,index,firstc2,ncetis,x,y = redux(d)
d = D[l3]
awaken,inbox3,distancias,hangon,waiting,count,index,firstc3,ncetis,x,y = redux(d) 
d = D[l4]
awaken,inbox4,distancias,hangon,waiting,count,index,firstc4,ncetis,x,y = redux(d) 

plt.hist(inbox1, bins=bins, histtype='step', align='mid', color='teal',
        label='l1s', linewidth=2, density=True)
plt.hist(inbox2, bins=bins, histtype='step', align='mid',color='slateblue',
        label='l2s', linewidth=2, density=True)
plt.hist(inbox3, bins=bins, histtype='step', align='mid',color='firebrick',
        label='l3s', linewidth=2, density=True)
plt.hist(inbox4, bins=bins, histtype='step', align='mid',color='tomato',
        label='l4s', linewidth=2, density=True)

plt.xticks(np.arange(0, 12, 2))
plt.yscale('log')
plt.legend()
plt.show()
          

from statsmodels.distributions.empirical_distribution import ECDF

ecdf1 = ECDF(inbox1)                                                   
ecdf2 = ECDF(inbox2)                                                   
ecdf3 = ECDF(inbox3)                                                   
ecdf4 = ECDF(inbox4)                                                   


#=========================================================  ACUMULADA, Fig. 1
plt.plot(ecdf1.x+1, ecdf1.y, label='dense awakening, short lifetime ', linewidth=1, color='teal')
plt.plot(ecdf2.x+1, ecdf2.y, label='sparse awakening, long lifetime ', linewidth=2, color='slateblue',linestyle='--')
plt.plot(ecdf3.x+1, ecdf3.y, label='dense awakening, long lifetime  ', linewidth=2, color='firebrick') 
plt.plot(ecdf4.x+1, ecdf4.y, label='sparse awakening, short lifetime', linewidth=1, color='tomato', linestyle='--')
 
plt.xlim(1,120)
plt.xscale('log') 
plt.legend(loc=4)
plt.xlabel('Multiplicity of contacts + 1, M+1')
plt.ylabel('N(<M)/N')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.rcParams.update({'font.size': 12})
plt.show()   
#=============================================================================


firstc1 = np.array(firstc1)/1000.
firstc2 = np.array(firstc2)/1000.
firstc3 = np.array(firstc3)/1000.
firstc4 = np.array(firstc4)/1000.

#====================================================================== Fig 2
bins = np.arange(0, 250, 5)
plt.hist(firstc1, bins=bins, histtype='step', align='mid', linewidth=1, linestyle='-', color='teal', label='dense awakening, short lifetime ')
plt.hist(firstc2, bins=bins, histtype='step', align='mid', linewidth=2, linestyle='--', color='slateblue', label='sparse awakening, long lifetime ')
plt.hist(firstc3, bins=bins, histtype='step', align='mid', linewidth=2, linestyle='-', color='firebrick', label='dense awakening, long lifetime ')
plt.hist(firstc4, bins=bins, histtype='step', align='mid', linewidth=1, linestyle='--', color='tomato', label='sparse awakening, short lifetime')

#plt.xticks(np.arange(0, 12, 2))
plt.yscale('log')
plt.xlim(1,120)
plt.legend(loc=1)
plt.xlabel('waiting time for first contact (x10^3 yr)')
plt.ylabel('EPDF')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.rcParams.update({'font.size': 12})
plt.show()   
#=============================================================================

 










#==================================================================Fig.
### RATE OF NO CONTACT VS. TAU_A AND TAU_S

import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage


m = pickle.load( open('../dat/SKRU_07/matrix1_d3_SKRU07.pkl', "rb") )


fig, ax = plt.subplots()
im = ax.imshow(m, interpolation='nearest', 
        extent=[Amin,Amax,Smin,Smax], cmap=cm.viridis,
        origin='lower', aspect='auto')
plt.xticks(A[::3], [str(int(a)) for a in A[::3]])
plt.yticks(S[::3], [str(int(a)) for a in S[::3]])
plt.title('rate of no contact')
plt.xlabel('awakening rate')
plt.ylabel('survival rate')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()
plt.show() 
#===========================================================================












#====================================================================Fig.

m = pickle.load( open('../dat/SKRU_07/matrix1_d3_SKRU07.pkl', "rb") )

sigma=[2,2]
m2t_smoothed=ndimage.filters.gaussian_filter(m2t, sigma)
m2_smoothed=ndimage.filters.gaussian_filter(m2, sigma)

levels = list(np.arange(np.min(m2), np.max(m2), (np.max(m2)-np.min(m2))/20 ))

# algoritmos de interpolacion:
#https://stackoverflow.com/questions/34230108/smoothing-imshow-plot-with-matplotlib


fig, ax = plt.subplots()
im = ax.imshow(m2, extent=[Amin,Amax,Smin,Smax], interpolation='none',
        cmap=cm.rainbow, origin='lower', aspect='auto')  
CS = ax.contour(S, A, m2_smoothed, levels=levels, colors='k', linewidths=0.4,
        alpha=0.5)
        #extent=[Amin,Amax,Smin,Smax],
        #origin='lower')    

ax.clabel(CS, inline=1, fontsize=10)

plt.xticks(A[::3], [str(int(a)) for a in A[::3]])
plt.yticks(S[::3], [str(int(a)) for a in S[::3]])
plt.title('rate of contact at awakening')
plt.xlabel('awakening rate')
plt.ylabel('survival rate')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()
plt.show()  


#===========================================================================

levels = np.arange(0.1,1.0,0.1)
 


fig, ax = plt.subplots()

im = ax.imshow(m2, origin='lower', aspect='auto',
        #interpolation='kaiser',
        extent=[Amin,Amax,Smin,Smax],
        vmin=0, vmax=1.,
        cmap=cm.viridis)

CS = ax.contour(m2_smoothed, levels=levels, colors='k',
        extent=[Amin,Amax,Smin,Smax],
        linewidths=0.4, alpha=0.4)

ax.clabel(CS, list(CS.levels[5:]), inline=1, fontsize=10, fmt='%1.1f')

plt.xticks(A[::3], [str(int(a)) for a in A[::3]])
plt.yticks(S[::3], [str(int(a)) for a in S[::3]])
plt.title('rate of contact at awakening')
plt.xlabel('awakening rate (x1000 yr)')
plt.ylabel('survival rate (x1000 yr)')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()
plt.show()  
                  

#===========================================================================












 
########################################### Fig.::: distr. galctocentric distance

l = (D['tau_awakening']< 37000) &  (D['tau_survive']> 60000) & (D['D_max']==50000.)
d = D[l]
awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d)

inbox = np.array(inbox)

l0 = inbox == 0
l1 = inbox == 1
l2 = inbox == 2
l3 = inbox == 3

x = np.array(x)
y = np.array(y)
#
#plt.scatter(x[l1],y[l1],color='slateblue', s=0.1)
#plt.scatter(x[l2],y[l2],color='crimson', alpha=0.6, s=0.4)
#plt.scatter(x[l3],y[l3],color='forestgreen',alpha=0.2, s=1)
#plt.show()

x = np.array(x)
y = np.array(y)

dist0 = np.sqrt(x[l0]**2 + y[l0]**2)
dist1 = np.sqrt(x[l1]**2 + y[l1]**2)
dist2 = np.sqrt(x[l2]**2 + y[l2]**2)
dist3 = np.sqrt(x[l3]**2 + y[l3]**2)

plt.hist(dist0, bins=30, histtype='step', density=True, align='mid',color='slategrey',linewidth=3, label='all')
plt.hist(dist1, bins=30, histtype='step', density=True, align='mid',color='slateblue', label='1')
plt.hist(dist2, bins=30, histtype='step', density=True, align='mid',color='crimson', label='2')
plt.hist(dist3, bins=30, histtype='step', density=True, align='mid',color='forestgreen', label='3')
plt.title('radial distribution of cetis with contact')
plt.xlabel('gactocentric distance (lyr)')
plt.ylabel('PDE')
plt.legend(loc=2)
plt.show()
#===========================================================================





########################################### Fig.::: comunicacion bidireccional


########################################### Fig.::: perfil radial normalizado

D = pandas.read_csv('../dat/SKRU_02/params.csv')

l = (D['tau_awakening']< 17000) &  (D['tau_survive']> 10000)
d = D[l]
awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d)

inbox = np.array(inbox)

l0 = inbox == 0
l1 = inbox == 1
l2 = inbox == 2
l3 = inbox == 3

x = np.array(x)
y = np.array(y)

distt = np.sqrt(x**2 + y**2)
dist0 = np.sqrt(x[l0]**2 + y[l0]**2)
dist1 = np.sqrt(x[l1]**2 + y[l1]**2)
dist2 = np.sqrt(x[l2]**2 + y[l2]**2)
dist3 = np.sqrt(x[l3]**2 + y[l3]**2)

# normalizar:



dsts = [dist0, dist1, dist2, dist3]
ns = [300, 100, 100, 100]
lbl = ['No contact','1','2','3']
colors = ['slategrey','slateblue','crimson','forestgreen']
lwd = [3,1,1,1]

for d, n, lb, c, l in zip(dsts, ns, lbl, colors, lwd):
    bins = np.linspace(20000,60000,n)
    Hy, Hx = np.histogram(d, bins=bins, density=False)
    Hm = (Hx[1:] + Hx[:-1])/2
    Hty, Htx = np.histogram(distt, bins=bins, density=False)
    Hy = Hy / Hty
    #plt.step(Hx[:-1], Hy, color=c,linewidth=l, label=lb, where='pre')
    plt.scatter(Hx[:-1], Hy, color=c, s=3, alpha=0.5, label=lb)

    p=np.polyfit(Hm, Hy, 2)
    f=Hm**2*p[0] + Hm*p[1] + p[2]
    plt.plot(Hm, f, color=c, linewidth=1) 


plt.title('radial distribution of cetis with contact')
plt.xlabel('gactocentric distance (lyr)')
plt.ylabel('frac')
plt.xlim(20000, 60000)
plt.yscale('log')
plt.legend(loc=2) 
plt.show()

