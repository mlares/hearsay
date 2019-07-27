import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(-30, 30), ylim=(-30, 30))
fill, = ax.plot([], [], lw=2)

def init():
    fill.set_data([], [])
    fill.set_color('cadetblue')
    return fill,

def animate(i):
    n = 100
    radii = [i/30.,3+i/30.]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    fill.set_data(np.ravel(xs), np.ravel(ys))
    fill.set_color('cadetblue')
#            color='cadetblue',edgecolor='green', alpha=0.3) 
    return fill,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=5, blit=True)

anim.save('anim.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
 
