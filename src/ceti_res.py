# coding: utf-8

import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

# local modules
from ceti_exp import redux


# Definition of subsamples:

recompute = False

if(recompute):
    la_low = D['tau_awakening'].isin([12000,16000,20000,24000,28000])
    la_upp = D['tau_awakening'].isin([172000,176000,180000,184000,188000])

    ls_low = D['tau_survive'].isin([10000,20000,30000,40000,50000])
    ls_upp = D['tau_survive'].isin([400000,410000,420000,430000,440000])

    ld_1k = D['D_max']==1000.
    ld_10k = D['D_max']==10000.
    ld_40k = D['D_max']==40000.
    ld_80k = D['D_max']==80000. 

#- - - -
    l1_d10k = la_low & ls_low & ld_10k
    l2_d10k = la_low & ls_upp & ld_10k
    l3_d10k = la_upp & ls_low & ld_10k
    l4_d10k = la_upp & ls_upp & ld_10k

    l1_d40k = la_low & ls_low & ld_40k
    l2_d40k = la_low & ls_upp & ld_40k
    l3_d40k = la_upp & ls_low & ld_40k
    l4_d40k = la_upp & ls_upp & ld_40k

    l1_d80k = la_low & ls_low & ld_80k
    l2_d80k = la_low & ls_upp & ld_80k
    l3_d80k = la_upp & ls_low & ld_80k
    l4_d80k = la_upp & ls_upp & ld_80k
#- - - -
     
    d = D[l1_d40k]
    awaken,inbox1,distancias,hangon,waiting,count,index,firstc1,ncetis,x,y = redux(d) 
    d = D[l2_d40k]
    awaken,inbox2,distancias,hangon,waiting,count,index,firstc2,ncetis,x,y = redux(d) 
    d = D[l3_d40k]
    awaken,inbox3,distancias,hangon,waiting,count,index,firstc3,ncetis,x,y = redux(d) 
    d = D[l4_d40k]
    awaken,inbox4,distancias,hangon,waiting,count,index,firstc4,ncetis,x,y = redux(d) 
     
    ecdf1 = ECDF(inbox1)                                                   
    ecdf2 = ECDF(inbox2)                                                   
    ecdf3 = ECDF(inbox3)                                                   
    ecdf4 = ECDF(inbox4)                                                   


#====================================================================== Fig. 4
def fig4():
    #{{{
 
# 1 = a_low & s_low  #'dense awakening, short lifetime '
# 2 = a_low & s_upp  #'dense awakening, long lifetime  '
# 3 = a_upp & s_low  #'sparse awakening, short lifetime'
# 4 = a_upp & s_upp  #'sparse awakening, long lifetime '


    plt.plot(ecdf1.x+1, ecdf1.y, label='dense awakening, short lifetime ',\
            linewidth=1, color='teal')
    plt.plot(ecdf2.x+1, ecdf2.y, label='dense awakening, long lifetime  ',\
            linewidth=2, color='slateblue',linestyle='--')
    plt.plot(ecdf3.x+1, ecdf3.y, label='sparse awakening, short lifetime',\
            linewidth=2, color='firebrick') 
    plt.plot(ecdf4.x+1, ecdf4.y, label='sparse awakening, long lifetime ',\
            linewidth=1, color='tomato', linestyle='--')
     
    plt.xlim(1,120)
    plt.xscale('log') 
    plt.legend(loc=4)
    plt.xlabel('Multiplicity of contacts + 1, M+1')
    plt.ylabel('N(<M)/N')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.rcParams.update({'font.size': 12})

    plt.savefig("foo.pdf", bbox_inches='tight')


#}}}

#====================================================================== Fig. 5
def fig5():
    #{{{

# 1 = a_low & s_low  #'dense awakening, short lifetime')
# 2 = a_low & s_upp  #'dense awakening, long lifetime')
# 3 = a_upp & s_low  #'sparse awakening, short lifetime')
# 4 = a_upp & s_upp  #'sparse awakening, long lifetime')

    firstc_n1 = np.array(firstc1)/1000.
    firstc_n2 = np.array(firstc2)/1000.
    firstc_n3 = np.array(firstc3)/1000.
    firstc_n4 = np.array(firstc4)/1000.

    bins = np.arange(0, 250, 10)
    plt.hist(firstc_n1, bins=bins, histtype='step', align='mid', linewidth=1, linestyle='-', color='teal', label='dense awakening, short lifetime ')
    plt.hist(firstc_n2, bins=bins, histtype='step', align='mid', linewidth=2, linestyle='--', color='slateblue', label='dense awakening, long lifetime')
    plt.hist(firstc_n3, bins=bins, histtype='step', align='mid', linewidth=2, linestyle='-', color='firebrick', label='sparse awakening, short lifetime')
    plt.hist(firstc_n4, bins=bins, histtype='step', align='mid', linewidth=1, linestyle='--', color='tomato', label='sparse awakening, long lifetime')

    plt.yscale('log')
    plt.xlim(1,250)
    plt.legend(loc=1)
    plt.xlabel('waiting time for first contact (x10^3 yr)')
    plt.ylabel('EPDF')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.rcParams.update({'font.size': 12})
    plt.show()   
    #plt.savefig("foo.pdf", bbox_inches='tight')
    #=============================================================================

     
#}}}



#====================================================================== Fig. 7
# 2D plots:
# rate of cETIs that never make contacts (plot_type=1)
# rate of CETIs that make contact at awakening (plot_type=2)
def fig7(experiment_run, plot_type, param_dmax):
    #{{{

    #experiment_run, plot_type, param_dmax = 'SKRU_07', 1, 2

    import pandas
    import pickle

    filename = '../dat/' + experiment_run + '/params_' + experiment_run + '.csv'
    with open(filename) as f:
        D = pandas.read_csv(f)
    
    filename = '../dat/' + experiment_run + '/matrix' + str(plot_type) \
               + '_d' + str(param_dmax) + '_' + experiment_run + '.pkl'
    print(filename)
    with open(filename,'rb') as f:
        m = pickle.load( f )

    # matrices 1: fraccion de cetis que no hacen nunca contacto
    # matrices 2: fraccion de cetis que hacen contacto en el awakening
    if plot_type == 1:
        titulo = 'rate of CETIs that never make contact'
    elif plot_type == 2:
        titulo = 'rate of contact at awakeninig'
    else:
        print('please use a valid plot_type')

    mt = np.transpose(m)
    sigma=[2,2]
    mt_smoothed=ndimage.filters.gaussian_filter(mt, sigma)
    m_smoothed=ndimage.filters.gaussian_filter(m, sigma)

    A = D['tau_awakening'].unique()
    A.sort()
    A = A / 1000.
    N1 = len(A)
    Amin = min(A)
    Amax = max(A)
    dA = (Amax-Amin)/N1
    Amin = Amin - dA/2
    Amax = Amax + dA/2

    S = D['tau_survive'].unique()
    S.sort()
    S = S / 1000.
    N2 = len(S)
    Smin = min(S)
    Smax = max(S)
    dS = (Smax-Smin)/N2
    Smin = Smin - dS/2
    Smax = Smax + dS/2
     

    levels = list(np.arange(np.min(m), np.max(m), (np.max(m)-np.min(m))/20 ))

    fig, ax = plt.subplots()

    im = ax.imshow(mt, origin='lower', aspect='auto',
            #interpolation='kaiser',
            extent=[Amin,Amax,Smin,Smax],
            vmin=0, vmax=1.,
            cmap=cm.viridis)
 
    CS = ax.contour(mt_smoothed, levels=levels, colors='k',
            extent=[Amin,Amax,Smin,Smax],
            linewidths=0.5, alpha=1.)

    ax.clabel(CS, list(CS.levels[5:]), inline=1, fontsize=10, fmt='%1.1f')

    plt.xticks(A[::7], [str(int(a)) for a in A[::7]])
    plt.yticks(S[::7], [str(int(a)) for a in S[::7]])


    plt.title(titulo)
    plt.xlabel('awakening rate [kyr]')
    plt.ylabel('survival rate [kyr]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.tight_layout()
    plt.show()  
#}}}


# Distribution of galctocentric distances
def plot7():
    #{{{

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
#}}}


# Normalized radial profile
def plot8():
    #{{{

    D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.csv')

    d = D[l1_d40k]
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
#}}}







def waiting_a_acum_log(reload=False):
    #{{{
 
    if(reload):
        D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.csv')

        ls = D['tau_survive'].isin([170000, 180000, 190000, 200000, 
                                    210000, 220000, 230000, 240000])
        ld = D['D_max'].isin([40000.])

        ls1 = ls & ld &   D['tau_awakening'].isin([8000  ])
        ls2 = ls & ld &   D['tau_awakening'].isin([28000 ])
        ls3 = ls & ld &   D['tau_awakening'].isin([48000 ])
        ls4 = ls & ld &   D['tau_awakening'].isin([68000 ])
        ls5 = ls & ld &   D['tau_awakening'].isin([108000])
        ls6 = ls & ld &   D['tau_awakening'].isin([188000])

        res1 = reddux(D[ls1])
        res2 = reddux(D[ls2])
        res3 = reddux(D[ls3])
        res4 = reddux(D[ls4])
        res5 = reddux(D[ls5])
        res6 = reddux(D[ls6])        

        ecdf1 = ECDF(res1['w'])
        ecdf2 = ECDF(res2['w'])
        ecdf3 = ECDF(res3['w'])
        ecdf4 = ECDF(res4['w'])
        ecdf5 = ECDF(res5['w'])
        ecdf6 = ECDF(res6['w'])

    plt.plot(ecdf1.x, ecdf1.y,   label='tau_a=8000')
    plt.plot(ecdf2.x, ecdf2.y,   label='28000') 
    plt.plot(ecdf3.x, ecdf3.y,   label='48000') 
    plt.plot(ecdf4.x, ecdf4.y,   label='68000') 
    plt.plot(ecdf5.x, ecdf5.y,   label='88000') 
    plt.plot(ecdf6.x, ecdf6.y,   label='108000')

    plt.title('D_max=40000, tau_s in [170000, 240000]')
    plt.xlabel('t (yr)')
    plt.ylabel('frac')
    plt.xlim(1.e1, 1.e6)
    plt.xscale('log')
    plt.legend(loc=4) 
    plt.show()
#}}}

def waiting_a_acum_lin(reload=False):
    #{{{
 
    if(reload):
        D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.csv')

        ls = D['tau_survive'].isin([170000, 180000, 190000, 200000, 
                                    210000, 220000, 230000, 240000])
        ld = D['D_max'].isin([40000.])

        ls1 = ls & ld &   D['tau_awakening'].isin([8000  ])
        ls2 = ls & ld &   D['tau_awakening'].isin([28000 ])
        ls3 = ls & ld &   D['tau_awakening'].isin([48000 ])
        ls4 = ls & ld &   D['tau_awakening'].isin([68000 ])
        ls5 = ls & ld &   D['tau_awakening'].isin([108000])
        ls6 = ls & ld &   D['tau_awakening'].isin([188000])

        res1 = reddux(D[ls1])
        res2 = reddux(D[ls2])
        res3 = reddux(D[ls3])
        res4 = reddux(D[ls4])
        res5 = reddux(D[ls5])
        res6 = reddux(D[ls6])        

        ecdf1 = ECDF(res1['w'])
        ecdf2 = ECDF(res2['w'])
        ecdf3 = ECDF(res3['w'])
        ecdf4 = ECDF(res4['w'])
        ecdf5 = ECDF(res5['w'])
        ecdf6 = ECDF(res6['w'])

    plt.plot(ecdf1.x, ecdf1.y,   label='tau_a=8000')
    plt.plot(ecdf2.x, ecdf2.y,   label='28000') 
    plt.plot(ecdf3.x, ecdf3.y,   label='48000') 
    plt.plot(ecdf4.x, ecdf4.y,   label='68000') 
    plt.plot(ecdf5.x, ecdf5.y,   label='88000') 
    plt.plot(ecdf6.x, ecdf6.y,   label='108000')

    plt.title('D_max=40000, tau_s in [170000, 240000]')
    plt.xlabel('t (yr)')
    plt.ylabel('frac')
    plt.xlim(1.e1, 1.e6)
    plt.legend(loc=4) 
    plt.show()
#}}}

def waiting_a_dif_log(reload=False):
    #{{{

    if(reload):
        D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.csv')

        ls = D['tau_survive'].isin([170000, 180000, 190000, 200000, 
                                    210000, 220000, 230000, 240000])
        ld = D['D_max'].isin([40000.])

        ls1 = ls & ld &   D['tau_awakening'].isin([8000  ])
        ls2 = ls & ld &   D['tau_awakening'].isin([28000 ])
        ls3 = ls & ld &   D['tau_awakening'].isin([48000 ])
        ls4 = ls & ld &   D['tau_awakening'].isin([68000 ])
        ls5 = ls & ld &   D['tau_awakening'].isin([108000])
        ls6 = ls & ld &   D['tau_awakening'].isin([188000])

        res1 = reddux(D[ls1])
        res2 = reddux(D[ls2])
        res3 = reddux(D[ls3])
        res4 = reddux(D[ls4])
        res5 = reddux(D[ls5])
        res6 = reddux(D[ls6])

    #bins = np.linspace(0,900000, 100, endpoint=False)
    bins=np.linspace(0,900, 50, endpoint=False)**2
    h1y , h1x  = np.histogram(res1['w'], bins=bins)
    h2y , h2x  = np.histogram(res2['w'], bins=bins)
    h3y , h3x  = np.histogram(res3['w'], bins=bins)
    h4y , h4x  = np.histogram(res4['w'], bins=bins)
    h5y , h5x  = np.histogram(res5['w'], bins=bins)
    h6y , h6x  = np.histogram(res6['w'], bins=bins)

    plt.step(h1x[:-1], h1y, alpha=0.5, label='tau_a=8000  ', where='pre')
    plt.step(h2x[:-1], h2y, alpha=0.5, label='28000 ', where='pre')
    plt.step(h3x[:-1], h3y, alpha=0.5, label='48000 ', where='pre')
    plt.step(h4x[:-1], h4y, alpha=0.5, label='68000 ', where='pre')
    plt.step(h5x[:-1], h5y, alpha=0.5, label='108000', where='pre')
    plt.step(h6x[:-1], h6y, alpha=0.5, label='188000', where='pre')


    plt.title('D_max=40000, tau_s in [170000, 240000]')
    plt.xlabel('t (yr)')
    plt.ylabel('frac')
    plt.xlim(0, 800000)
    plt.yscale('log')
    plt.legend(loc=1) 
    plt.show()
#}}}




def waiting_s_acum_log(reload=False):
    #{{{
 
    if(reload):
        D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.csv')

        la = D['tau_awakening'].isin([68000,  72000,  76000,  80000,  84000,  88000,  
                                      92000,  96000, 100000])
        ld = D['D_max'].isin([40000.])

        ls1 = la & ld &   D['tau_survive'].isin([10000 ])
        ls2 = la & ld &   D['tau_survive'].isin([100000])
        ls3 = la & ld &   D['tau_survive'].isin([200000])
        ls4 = la & ld &   D['tau_survive'].isin([300000])
        ls5 = la & ld &   D['tau_survive'].isin([400000])

        res1 = reddux(D[ls1])
        res2 = reddux(D[ls2])
        res3 = reddux(D[ls3])
        res4 = reddux(D[ls4])
        res5 = reddux(D[ls5])

        ecdf1 = ECDF(res1['w'])
        ecdf2 = ECDF(res2['w'])
        ecdf3 = ECDF(res3['w'])
        ecdf4 = ECDF(res4['w'])
        ecdf5 = ECDF(res5['w'])

    plt.plot(ecdf1.x, ecdf1.y,   label='tau_s=10kyr')
    plt.plot(ecdf2.x, ecdf2.y,   label='100kyr') 
    plt.plot(ecdf3.x, ecdf3.y,   label='200kyr') 
    plt.plot(ecdf4.x, ecdf4.y,   label='300kyr') 
    plt.plot(ecdf5.x, ecdf5.y,   label='400kyr') 

    plt.title('D_max=40000, tau_a in [68000,100000]')
    plt.xlabel('t (yr)')
    plt.ylabel('frac')
    plt.xlim(1.e1, 1.e6)
    plt.xscale('log')
    plt.legend(loc=4) 
    plt.show()
#}}}

def waiting_s_acum_lin(reload=False):
    #{{{
 
    if(reload):
        D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.csv')

        la = D['tau_awakening'].isin([68000,  72000,  76000,  80000,  84000,  88000,  
                                      92000,  96000, 100000])
        ld = D['D_max'].isin([40000.])

        ls1 = la & ld &   D['tau_survive'].isin([10000 ])
        ls2 = la & ld &   D['tau_survive'].isin([100000])
        ls3 = la & ld &   D['tau_survive'].isin([200000])
        ls4 = la & ld &   D['tau_survive'].isin([300000])
        ls5 = la & ld &   D['tau_survive'].isin([400000])

        res1 = reddux(D[ls1])
        res2 = reddux(D[ls2])
        res3 = reddux(D[ls3])
        res4 = reddux(D[ls4])
        res5 = reddux(D[ls5])

        ecdf1 = ECDF(res1['w'])
        ecdf2 = ECDF(res2['w'])
        ecdf3 = ECDF(res3['w'])
        ecdf4 = ECDF(res4['w'])
        ecdf5 = ECDF(res5['w'])

    plt.plot(ecdf1.x, ecdf1.y,   label='tau_s=10kyr')
    plt.plot(ecdf2.x, ecdf2.y,   label='100kyr') 
    plt.plot(ecdf3.x, ecdf3.y,   label='200kyr') 
    plt.plot(ecdf4.x, ecdf4.y,   label='300kyr') 
    plt.plot(ecdf5.x, ecdf5.y,   label='400kyr') 

    plt.title('D_max=40000, tau_a in [68000,100000]')
    plt.xlabel('t (yr)')
    plt.ylabel('frac')
    plt.xlim(1.e1, 1.e6)
    plt.legend(loc=4) 
    plt.show()
#}}}


def waiting_s_dif(reload=False):
    #{{{

    if(reload):
        D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.csv')

        la = D['tau_awakening'].isin([68000,  72000,  76000,  80000,  84000,  88000,  
                                      92000,  96000, 100000])
        ld = D['D_max'].isin([40000.])

        ls1 = la & ld &   D['tau_survive'].isin([10000 ])
        ls2 = la & ld &   D['tau_survive'].isin([100000])
        ls3 = la & ld &   D['tau_survive'].isin([200000])
        ls4 = la & ld &   D['tau_survive'].isin([300000])
        ls5 = la & ld &   D['tau_survive'].isin([400000])

        res1 = reddux(D[ls1])
        res2 = reddux(D[ls2])
        res3 = reddux(D[ls3])
        res4 = reddux(D[ls4])
        res5 = reddux(D[ls5])           


    #bins = np.linspace(0,900000, 100, endpoint=False)
    bins=np.linspace(0,900, 20, endpoint=False)**2
    h1y , h1x  = np.histogram(res1['w'], bins=bins)
    h2y , h2x  = np.histogram(res2['w'], bins=bins)
    h3y , h3x  = np.histogram(res3['w'], bins=bins)
    h4y , h4x  = np.histogram(res4['w'], bins=bins)
    h5y , h5x  = np.histogram(res5['w'], bins=bins)

    plt.step(h1x[:-1], h1y, alpha=0.5, label='tau_s=10kyr', where='pre')
    plt.step(h2x[:-1], h2y, alpha=0.5, label='100kyr', where='pre')
    plt.step(h3x[:-1], h3y, alpha=0.5, label='200kyr', where='pre')
    plt.step(h4x[:-1], h4y, alpha=0.5, label='300kyr', where='pre')
    plt.step(h5x[:-1], h5y, alpha=0.5, label='400kyr', where='pre')


    plt.title('D_max=40000, tau_a in [68000,100000]')
    plt.xlabel('t (yr)')
    plt.ylabel('frac')
    plt.xlim(0, 800000)
    plt.yscale('log')
    plt.legend(loc=1) 
    plt.show()
#}}}


