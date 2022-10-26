import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
import numpy as np
from scipy.linalg import orthogonal_procrustes


def create_intermediate_matrices(x_i, x_f, n=5):
    """
    Creates an intermediate list of matrices to animate an orthogonal transformation of `x_i` into `x_f`.
    Warning: it will not work if `x_f` is reflected.

    Args:
        x_i(np.array) : The initial matrix.
        x_f(np.array) : The final matrix.
        n(int) : Number of intermediate matrices to generate.
    
    Returns:
        x_list(list[np.array]) : The list of `n` intermediate matrices.
    """

    theta = np.dot(x_i[0], x_f[0])/(np.linalg.norm(x_i[0]) * np.linalg.norm(x_f[0]))
    theta = np.arccos(theta)  # Final rotation angle
    print("Theta", theta)
    x_list = list()
    # x = x_i
    # t_angle = theta/n
    for t in range(n, 0, -1):
        t_angle = theta/t
        print("angle", t_angle)
        r = np.array([[np.cos(t_angle), -np.sin(t_angle)], [np.sin(t_angle), np.cos(t_angle)]])
        x = np.dot(x_i, r)
        x_list.append(x)

    return x_list  


np.random.seed(1)
plt.rcParams.update({'font.size': 12})
sns.set_style("whitegrid")

n=300
alpha=0.75
n_landmarks = 20
color_1 = 'yellowgreen'
color_2 = 'rebeccapurple'
color_accent = 'coral'
label_A = 'A'
label_B = 'B'
x_ticks = [-1, 0, 1]
y_ticks = [-1, 0, 1]
xlim = (-1.0, 1.0)
ylim = (-1.0, 1.0)
x1 = np.random.normal((0,0), (0.3,0.3), size=(n, 2))
theta = 0.5*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# Rotate base embedding and add some noise
x2 = np.dot(x1, R)
x2 += np.random.normal((0, 0), (0.01, 0.05), size=(n, 2))

landmarks_idx = np.random.choice(range(len(x1)), size=(n_landmarks))

fig, ax = plt.subplots()

for i, p in enumerate(x1):
    ax.scatter(*p, color=color_1, alpha=alpha, label=label_A if i == 0 else '')
for i, p in enumerate(x2):
    ax.scatter(*p, color=color_2, alpha=alpha, label=label_B if i == 0 else '')

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.legend()
fig.savefig('embeddings_before.pdf')
plt.close()

# Landmarks

fig, ax = plt.subplots()
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
for i, p in enumerate(x1):
    ax.scatter(*p, color=color_1, alpha=alpha, label=label_A if i == 0 else '')
for i, p in enumerate(x2):
    ax.scatter(*p, color=color_2, alpha=alpha, label=label_B if i == 0 else '')

for i, idx in enumerate(landmarks_idx):
    ax.scatter(*x1[idx], edgecolors=color_accent, facecolors='none', linewidth=2, label='Landmark' if i == 0 else '')
    ax.scatter(*x2[idx], edgecolors=color_accent, facecolors='none', linewidth=2)
ax.legend(loc='upper right')
fig.savefig('embeddings_landmarks.pdf')

for i, idx in enumerate(landmarks_idx):
    con = ConnectionPatch(x1[idx], x2[idx], coordsA='data', coordsB='data', shrinkA=0, shrinkB=0, edgecolor='black', linewidth=2)
    ax.add_patch(con)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.legend(loc='upper right')
fig.savefig('embeddings_landmarks_connected.pdf')
plt.close()

# After alignment
Q, _ = orthogonal_procrustes(x1[landmarks_idx], x2[landmarks_idx])
x1_i = np.copy(x1)
x1 = np.dot(x1, Q)
det = np.linalg.det(Q).round(decimals=0)  # If det=1, `Q` only performs rotation.
print(Q)
q_theta = np.arccos(Q[0][0])

fig, ax = plt.subplots()
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
for i, p in enumerate(x1):
    ax.scatter(*p, color=color_1, alpha=alpha, label=label_A if i == 0 else '')
for i, p in enumerate(x2):
    ax.scatter(*p, color=color_2, alpha=alpha, label=label_B if i == 0 else '')

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.legend()
fig.savefig('embeddings_aligned.pdf')

for i, idx in enumerate(landmarks_idx):
    ax.scatter(*x1[idx], edgecolors=color_accent, facecolors='none', linewidth=2, label='Landmark' if i == 0 else '')
    ax.scatter(*x2[idx], edgecolors=color_accent, facecolors='none', linewidth=2)
    con = ConnectionPatch(x1[idx], x2[idx], coordsA='data', coordsB='data', shrinkA=0, shrinkB=0, edgecolor='black', linewidth=2)
    ax.add_patch(con)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.legend(loc='upper right')
fig.savefig('embeddings_aligned_landmarks.pdf')
plt.close()

# Animation

fig, ax = plt.subplots()
n_frames = 100
theta_frames = np.linspace(0, q_theta, n_frames)  # Intermediate angles
theta_frames = np.append(theta_frames, [q_theta]*(n_frames//2))

# for i, p in enumerate(x1):
#     ax.scatter(*p, color=color_1, alpha=alpha, label=label_A if i == 0 else '')
# for i, p in enumerate(x2):
#     ax.scatter(*p, color=color_2, alpha=alpha, label=label_B if i == 0 else '')
scat_1 = ax.scatter(x1_i[:, 0], x1_i[:, 1], alpha=alpha, label=label_A, color=color_1)
scat_2 = ax.scatter(x2[:, 0], x2[:, 1], alpha=alpha, label=label_B, color=color_2)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.legend()

def animate_rotation(i):
    ax.cla()
    r = np.array([[np.cos(theta_frames[i]), -np.sin(theta_frames[i])], [np.sin(theta_frames[i]), np.cos(theta_frames[i])]])
    x = np.dot(x1_i, r)
    scat_1 = ax.scatter(x[:, 0], x[:, 1], alpha=alpha, color=color_1)
    scat_2 = ax.scatter(x2[:, 0], x2[:, 1], alpha=alpha, color=color_2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    for i, idx in enumerate(landmarks_idx):
        ax.scatter(*x[idx], edgecolors=color_accent, facecolors='none', linewidth=2)
        ax.scatter(*x2[idx], edgecolors=color_accent, facecolors='none', linewidth=2)
        con = ConnectionPatch(x[idx], x2[idx], coordsA='data', coordsB='data', shrinkA=0, shrinkB=0, edgecolor='black', linewidth=2)
        ax.add_patch(con)

    return scat_1,


ani = animation.FuncAnimation(fig, animate_rotation, range(len(theta_frames)), interval=20, blit=True)
writervideo = animation.FFMpegWriter(fps=60)
f=r"rotation.gif"
ani.save(f)