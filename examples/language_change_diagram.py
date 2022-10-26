import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import numpy as np


"""
Produces diagrams showing language evolution in over time and cross domain directins.
"""
plt.rcParams.update({"font.size": 16})

fig, ax = plt.subplots()

# Over time
x0 = (-1, 0)
x1 = (1, 0)

cmap = matplotlib.cm.get_cmap('Paired')
colors = [cmap(i) for i in range(10)]

c1=colors[2]
c2=colors[3]
markerlinewidth=2
markersize=64**2

offset_x = -64
offset_y = 40

ax.scatter(*x0, s=markersize, color=c1, linewidth=markerlinewidth, edgecolors='black')
ax.scatter(*x1, s=markersize, color=c2, linewidth=markerlinewidth, edgecolors='black')
con = ConnectionPatch(x0, x1, coordsA='data', coordsB='data', shrinkA=36, shrinkB=36, 
                      edgecolor='black', linewidth=2, arrowstyle='->')
ax.add_patch(con)

# Add connection label 
ax.text(np.mean((x0[0], x1[0])), np.mean((x0[1], x1[1]))-0.1, "$\delta_t$",
        horizontalalignment='center', verticalalignment='center')

ax.annotate('English in 1800s', x0, xycoords='data', textcoords='offset points', 
            xytext=(offset_x, offset_y))
ax.annotate('English in 2000s', x1, xycoords='data', textcoords='offset points', 
            xytext=(offset_x, offset_y))

ax.text(x0[0], x0[1], '$L(t_0)$', horizontalalignment='center', verticalalignment='center')
ax.text(x1[0], x1[1], '$L(t_1)$', horizontalalignment='center', verticalalignment='center')

ax.set_xlim(-1.33, 1.33)
ax.set_ylim(-1, 1)
plt.axis('off')
plt.tight_layout()
plt.savefig('change_over_time.pdf')

plt.close()

### PLOT CROSS DOMAIN DIAGRAM

fig, ax = plt.subplots()

x0 = (0, 1)
xa = (-1, 0)
xb = (1, 0)

cmap = matplotlib.cm.get_cmap('Accent')
colors = [cmap(i) for i in range(10)]
c0 = colors[0]
ca = colors[1]
cb = colors[2]

curve_angle=0.33

ax.scatter(*x0, s=markersize, color=c0, linewidth=markerlinewidth, edgecolors='black')
ax.scatter(*xa, s=markersize, color=ca, linewidth=markerlinewidth, edgecolors='black')
ax.scatter(*xb, s=markersize, color=cb, linewidth=markerlinewidth, edgecolors='black')

con_a = ConnectionPatch(x0, xa, coordsA='data', coordsB='data', shrinkA=36, shrinkB=36, 
                      edgecolor='black', linewidth=2, arrowstyle='->', connectionstyle='arc3')
con_b = ConnectionPatch(x0, xb, coordsA='data', coordsB='data', shrinkA=36, shrinkB=36, 
                      edgecolor='black', linewidth=2, arrowstyle='->')

con_a.set_connectionstyle('Arc3',rad=curve_angle)
con_b.set_connectionstyle('Arc3',rad=-curve_angle)

ax.add_patch(con_a)
ax.add_patch(con_b)

# Marker labels
ax.text(x0[0], x0[1], '$L$', horizontalalignment='center', verticalalignment='center')
ax.text(xa[0], xa[1], '$L(a)$', horizontalalignment='center', verticalalignment='center')
ax.text(xb[0], xb[1], '$L(b)$', horizontalalignment='center', verticalalignment='center')

# Texts
ax.text(x0[0], x0[1]+0.28, 'English', horizontalalignment='center', verticalalignment='center')
ax.text(xa[0], xa[1]-0.28, 'British English', horizontalalignment='center', verticalalignment='center')
ax.text(xb[0], xb[1]-0.28, 'American English', horizontalalignment='center', verticalalignment='center')

# Add deltas
ax.text(np.mean((x0[0], xa[0]))+0.01, np.mean((x0[1], xa[1]))-0.01, "$\delta_a$",
        horizontalalignment='center', verticalalignment='center')
ax.text(np.mean((x0[0], xb[0]))-0.01, np.mean((x0[1], xb[1]))-0.01, "$\delta_b$",
        horizontalalignment='center', verticalalignment='center')

ax.set_xlim(-1.33, 1.33)
ax.set_ylim(-0.33, 1.33)
plt.axis('off')
plt.tight_layout()
plt.savefig('change_cross_domain.pdf')
