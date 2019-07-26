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
    ys = np.outer(radii, np.sin(theta)) + shift[0]
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    ax.fill(np.ravel(xs), np.ravel(ys), 
            color='cadetblue',edgecolor='green', alpha=0.3)

plt.show()

# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
# https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
# https://plot.ly/python/animations/
