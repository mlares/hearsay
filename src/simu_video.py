import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

patches = []

patches += [
    Wedge((.7,.8), .2, 0, 360, width=0.05),
    Wedge((.9,.3), .4, 0, 360, width=0.05),
    Wedge((.6,.5), .1, 0, 360, width=0.05),
]

ax = plt.subplot(111, aspect='equal')

fcolors = [10]*len(patches)
ecolors = [0]*len(patches)

p = PatchCollection(patches, 
        cmap=matplotlib.cm.jet, alpha=0.4)

p.set_array(np.array(fcolors))
ax.add_collection(p)
plt.show()






import numpy as np
import matplotlib.pyplot as plt


n = 100.
theta = np.linspace(0, 2*np.pi, n, endpoint=True)



ax = plt.subplot(111, aspect='equal')

rads = [[1., 1.9],[2.2, 3.45], [1.3, 2.7]]
shifts = [[0,0],[1,3],[-2,1]]

for radii, shift in zip(rads, shifts):
    xs = np.outer(radii, np.cos(theta)) + shift[0]
    ys = np.outer(radii, np.sin(theta)) + shift[1]
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    ax.fill(np.ravel(xs), np.ravel(ys), 
            color='cadetblue',edgecolor='green', alpha=0.3)

plt.xlim([-10,10])
plt.ylim([-10,10])
plt.show()





# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
# https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
# https://plot.ly/python/animations/


import pickle
import numpy as np
CETIs = pickle.load( open('dat/SKRU_08/D50000/01500_012.dat', "rb") )


N = len(CETIs)


x = []
y = []
t_a =[]
t_d =[]

for i in range(N):
    x.append(CETIs[i][0][2])
    y.append(CETIs[i][0][3])
    t_a.append(CETIs[i][0][4])
    t_d.append(CETIs[i][0][5])
#------------------------------


x = [0, 10, 15, 30]
y = [0, 0, 0, 0]
t_a = [1, 7, 12, 20]
t_d = [11, 12, 19, 24]
Dmax = 10.

x = np.array(x)
y = np.array(y)
t_a = np.array(t_a)
t_d = np.array(t_d)


# generate all the frames to make a nice video

tmax = max(max(t_a),max(t_d))
Nt = 100
time_steps = np.linspace(0,tmax,Nt) 


for t in time_steps:

   dt_a = t_a - t
   dt_d = t_d - t

   dt_a[dt_a<0] = None
   r1 = np.minimum(dt_a, Dmax)

   dt_d[dt_d<0] = None
   r2 = np.minimum(dt_d, Dmax)    

   print(t)
   print(r1)
   print(r2)
   print('--------------------')








matplotlib.use("Agg")
import matplotlib.animation as animation


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

# Set up formatting for the movie files
Writer = animation.writers['FFMpegWriter']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                   interval=50, blit=True)
line_ani.save('lines.mp4', writer=writer)

fig2 = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
base = np.hypot(x, y)
ims = []
for add in np.arange(15):
    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                   blit=True)
im_ani.save('im.mp4', writer=writer)







# Third-party
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
from JSAnimation.IPython_display import display_animation


n_frames = 128
n_trails = 8


t = np.linspace(0, 10, n_frames)
x = np.sin(t)
y = np.cos(t)


fig,ax = plt.subplots(1,1,figsize=(8,8))

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

# turn off axis spines
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# set figure background opacity (alpha) to 0
fig.patch.set_alpha(0.)

fig.tight_layout()

pt, = ax.plot([], [], linestyle='none', marker='o', ms=15, color='r')

trails = []
for i,alpha in enumerate(np.linspace(1.,0,n_trails)):
    l, = ax.plot([], [], linestyle='none', marker='o', ms=6, alpha=alpha, c='w', zorder=-1000)
    trails.append(l)

def init():
    pt.set_data([], [])
    for trail in trails:
        trail.set_data([], [])
    return (pt,) + tuple(trails)

def update(i):
    ix = i - n_trails

    pt.set_data(x[i], y[i])
    for j,trail in zip(range(len(trails))[::-1],trails):
        if ix+j < 0:
            continue
        trail.set_data(x[ix+j], y[ix+j])
    return (pt,) + tuple(trails)

ani = animation.FuncAnimation(fig, update, n_frames, init_func=init,
                              interval=20, blit=True)

display_animation(ani)























###########################################################################
###########################################################################
###########################################################################


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(-30, 30), ylim=(-30, 30))
fill, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    fill.set_data([], [])
    return fill,

# animation function.  This is called sequentially
def animate(i):
    n = 100
    radii = [i/10.,i+3/10.]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    fill.set_data(np.ravel(xs), np.ravel(ys))
    fill.color('cadetblue')
    #,color='cadetblue')
#            color='cadetblue',edgecolor='green', alpha=0.3) 
    return fill,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=10, blit=True)

anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


# Tutorial for animations:
# https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/


ver https://stackoverflow.com/questions/16120801/matplotlib-animate-fill-between-shape

###########################################################################
###########################################################################
###########################################################################
