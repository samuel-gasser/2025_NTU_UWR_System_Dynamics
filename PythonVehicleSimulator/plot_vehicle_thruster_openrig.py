#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------- Data & helpers ----------------------

# Thruster poses (position from CO and orientation angles)
thrusters = [
    # name,   x        y        z         phi   theta        psi
    ("vt1",  +0.1645, +0.254,  0.061339, 0.0,  -np.pi/2,    0.0),
    ("vt2",  -0.1645, +0.254,  0.061339, 0.0,  -np.pi/2,    0.0),
    ("vt3",  -0.1645, -0.254,  0.061339, 0.0,  -np.pi/2,    0.0),
    ("vt4",  +0.1645, -0.254,  0.061339, 0.0,  -np.pi/2,    0.0),
    ("ht1",  +0.150,  +0.150,  0.124,    0.0,   0.0,       -0.785398163),
    ("ht2",  -0.150,  +0.150,  0.124,    0.0,   0.0,       -2.35619449),
    ("ht3",  -0.150,  -0.150,  0.124,    0.0,   0.0,        2.35619449),
    ("ht4",  +0.150,  -0.150,  0.124,    0.0,   0.0,        0.785398163),
]
names = [t[0] for t in thrusters]
pos   = np.array([[t[1], t[2], t[3]] for t in thrusters])
euls  = np.array([[t[4], t[5], t[6]] for t in thrusters])

# Basic thrust directions:
# - VT: +z_b
# - HT: rotate +x by ψ around z (ϕ=θ=0)
f_dir = np.zeros((8,3))
for i, (name, *_rest) in enumerate(thrusters):
    if name.startswith("vt"):
        f_dir[i] = np.array([0.0, 0.0, 1.0])
    else:
        psi = euls[i][2]
        f_dir[i] = np.array([np.cos(psi), np.sin(psi), 0.0])

# Decoupling sign matrix (your final version)
# rows: [X, Y, Z, K, M, N] ; cols: vt1 vt2 vt3 vt4 ht1 ht2 ht3 ht4
S = np.array([
    [ 0,  0,  0,  0,  +1, -1, -1, +1],   # X (surge)
    [ 0,  0,  0,  0,  -1, -1, +1, +1],   # Y (sway)
    [ +1, +1, +1, +1,  0,  0,  0,  0],   # Z (heave)
    [ +1, +1, -1, -1,  0,  0,  0,  0],   # K (roll)
    [ -1, +1, +1, -1,  0,  0,  0,  0],   # M (pitch)
    [ 0,  0,  0,  0,  -1, +1, -1, +1],   # N (yaw)
], dtype=float)

dof_titles = [r"$X$ (surge)", r"$Y$ (sway)", r"$Z$ (heave)",
              r"$K$ (roll)", r"$M$ (pitch)", r"$N$ (yaw)"]

def set_axes_equal(ax):
    X = ax.get_xlim3d(); Y = ax.get_ylim3d(); Z = ax.get_zlim3d()
    max_range = max(abs(X[1]-X[0]), abs(Y[1]-Y[0]), abs(Z[1]-Z[0]))
    x_mid = np.mean(X); y_mid = np.mean(Y); z_mid = np.mean(Z)
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

# ---------------------- Plot 2×3 grid ----------------------

fig = plt.figure(figsize=(14, 9))  # wide figure for 2 rows × 3 cols
scale = 0.12
elev, azim = 30, -60  # "home" view

for i in range(6):
    ax = fig.add_subplot(2, 3, i+1, projection="3d")
    ax.set_title(f"Thruster basic vectors for decoupled movement: {dof_titles[i]}")
    ax.set_xlabel(r"$x_b$ [m]")
    ax.set_ylabel(r"$y_b$ [m]")
    ax.set_zlabel(r"$z_b$ [m]")
    ax.view_init(elev=elev, azim=azim)

    # Colored body axes with labels at origin
    origin = np.zeros(3)
    ax.quiver(*origin, 0.25, 0, 0, color="red",   length=1.0, normalize=False);  ax.text(0.27, 0, 0, r"$x_b$", color="red",   fontsize=10)
    ax.quiver(*origin, 0, 0.25, 0, color="green", length=1.0, normalize=False);  ax.text(0, 0.27, 0, r"$y_b$", color="green", fontsize=10)
    ax.quiver(*origin, 0, 0, 0.25, color="blue",  length=1.0, normalize=False);  ax.text(0, 0, 0.27, r"$z_b$", color="blue",  fontsize=10)

    # Thruster markers
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=18, c="#C09040")
    for j, name in enumerate(names):
        ax.text(pos[j,0], pos[j,1], pos[j,2], f" {name}", fontsize=8, color="#7a6a4a")

    # Active thrust vectors for this DOF
    signs = S[i]
    for j in range(8):
        if signs[j] == 0:
            continue
        v = signs[j] * f_dir[j] * scale
        ax.quiver(pos[j,0], pos[j,1], pos[j,2], v[0], v[1], v[2],
                  length=1.0, normalize=False, color="#E59700")  # thruster arrows

    # Consistent bounds
    ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.3, 0.3); ax.set_zlim(0.0, 0.25)
    set_axes_equal(ax)

plt.tight_layout()
plt.show()
