# -*- coding: utf-8 -*-
"""
Simulator plotting functions:

plotVehicleStates(simTime, simData, figNo) 
plotControls(simTime, simData, vehicle, figNo)
def plot3D(simData, numDataPoints, FPS, filename, figNo)

Author:     Thor I. Fossen
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from python_vehicle_simulator.lib.gnc import ssa
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Global plotting style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.2,
})


legendSize = 8  # legend size
figSize1 = [20, 10]  # figure1 size in cm
figSize2 = [20, 10]  # figure2 size in cm
dpiValue = 150  # figure dpi value


def R2D(value):  # radians to degrees
    return value * 180 / math.pi


def cm2inch(value):  # inch to cm
    return value / 2.54


# plotVehicleStates(simTime, simData, figNo) plots the 6-DOF vehicle
# position/attitude and velocities versus time in figure no. figNo
def plotVehicleStates(simTime, simData, figNo):
    # Time
    t = simTime

    # States (keep SI: m, rad, m/s, rad/s)
    x = simData[:, 0]              # East
    y = simData[:, 1]              # North
    z = simData[:, 2]              # Down (positive)
    phi   = ssa(simData[:, 3])
    theta = ssa(simData[:, 4])
    psi   = ssa(simData[:, 5])
    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]
    p = simData[:, 9]
    q = simData[:, 10]
    r = simData[:, 11]

    # Derived angles (rad) & speed
    U = np.sqrt(u*u + v*v + w*w)
    beta_c  = ssa(np.arctan2(v, u))                   # crab
    alpha_c = ssa(np.arctan2(w, u))                   # flight path
    chi     = ssa(simData[:, 5] + np.arctan2(v, u))   # course

    # Figure
    fig, axs = plt.subplots(
        3, 3, num=figNo,
        figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])),
        dpi=dpiValue, constrained_layout=True
    )
    fig.suptitle("Vehicle states")

    # Helper to prettify an axis
    def prettify(ax, xlabel=None, ylabel=None, y0=True, sci=False, ylim=None):
        ax.grid(True, linestyle=":", color="0.3", alpha=0.5)
        if y0: ax.axhline(0, color="0.3", lw=1, alpha=0.4)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if sci: ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        if ylim is not None: ax.set_ylim(*ylim)

   # (1,1) North–East track — same subplot size as others (no equal-aspect squeeze)
    ax = axs[0, 0]
    ax.set_title("xy plot")

    # Fossen positions: x = North, y = East
    ax.plot(x, y, color="C0")

    # Symmetric autoscale with padding and a minimum span so nearly-1D paths are visible
    pad = 0.08            # 8% padding
    min_span = 0.5        # [m] minimum side length so you see something

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = xmax - xmin
    dy = ymax - ymin
    span = max(dx, dy, min_span)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span * (1.0 + pad)

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    # Fill the subplot like the others (no square constraint)
    ax.set_aspect('auto')
    ax.set_anchor('C')   # keep the data box centered

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", color="0.3", alpha=0.5)
    ax.axhline(0, color="0.3", lw=1, alpha=0.35)
    ax.axvline(0, color="0.3", lw=1, alpha=0.35)

    # (1,2) Depth
    ax = axs[0,1]
    ax.plot(t, z)
    ax.set_title("Depth")
    prettify(ax, xlabel=None, ylabel="z (m)", y0=True)

    # (1,3) Roll & Pitch angles
    ax = axs[0,2]
    ax.plot(t, phi, label="Roll, Φ (rad)")
    ax.plot(t, theta, label="Pitch, θ (rad)")
    ax.set_title("Attitude (angles)")
    prettify(ax, xlabel=None, ylabel="rad", y0=True, sci=True)
    ax.legend(frameon=False, loc="best")

    # (2,1) Speed
    ax = axs[1,0]
    ax.plot(t, U)
    ax.set_title("Speed")
    prettify(ax, xlabel=None, ylabel="m/s", y0=True)

    # (2,2) Course angle
    ax = axs[1,1]
    ax.plot(t, chi)
    ax.set_title("Course")
    prettify(ax, xlabel=None, ylabel="rad", y0=True)
    ax.set_ylim(-np.pi, np.pi)

    # (2,3) Pitch & flight-path
    ax = axs[1,2]
    ax.plot(t, theta, label="Pitch, θ (rad)")
    ax.plot(t, alpha_c, label="Flight path (rad)")
    ax.set_title("Pitch vs. flight path")
    prettify(ax, xlabel=None, ylabel="rad", y0=True)
    ax.legend(frameon=False, loc="best")

    # (3,1) Linear velocities
    ax = axs[2,0]
    ax.plot(t, u, label="Surge, u (m/s)")
    ax.plot(t, v, label="Sway, v (m/s)")
    ax.plot(t, w, label="Heave, w (m/s)")
    prettify(ax, xlabel="Time (s)", ylabel="m/s", y0=True, sci=True)
    ax.legend(frameon=False, loc="best")

    # (3,2) Angular rates
    ax = axs[2,1]
    ax.plot(t, p, label="Roll rate, p (rad/s)")
    ax.plot(t, q, label="Pitch rate, q (rad/s)")
    ax.plot(t, r, label="Yaw rate, r (rad/s)")
    prettify(ax, xlabel="Time (s)", ylabel="rad/s", y0=True, sci=True)
    ax.legend(frameon=False, loc="best")

    # (3,3) Heading & crab
    ax = axs[2,2]
    ax.plot(t, psi,   label="Yaw, ψ (rad)")
    ax.plot(t, beta_c, label="Crab (rad)")
    prettify(ax, xlabel="Time (s)", ylabel="rad", y0=True)
    ax.legend(frameon=False, loc="best")




# plotControls(simTime, simData) plots the vehicle control inputs versus time
# in figure no. figNo
def plotControls(simTime, simData, vehicle, figNo):

    DOF = 6
    t = simTime

    plt.figure(figNo, figsize=(cm2inch(figSize2[0]), cm2inch(figSize2[1])), dpi=dpiValue)

    col = 2
    row = int(math.ceil(vehicle.dimU / col))

    for i in range(vehicle.dimU):

        u_control = simData[:, 2 * DOF + i]                  # command
        u_actual  = simData[:, 2 * DOF + vehicle.dimU + i]   # actual

        # If label mentions deg, convert series to radians and relabel to 'rad'
        label_i = vehicle.controls[i]
        if "deg" in label_i:
            u_control = np.deg2rad(u_control)
            u_actual  = np.deg2rad(u_actual)
            label_i = label_i.replace("deg", "rad")

        plt.subplot(row, col, i + 1)
        plt.plot(t, u_control, t, u_actual)
        plt.legend([label_i + " (command)", label_i + " (actual)"], fontsize=legendSize)
        plt.xlabel("Time (s)", fontsize=12)
        plt.grid()


# plot3D(simData,numDataPoints,FPS,filename,figNo) plots the vehicles position (x, y, z) in 3D
# in figure no. figNo
def euler2Cnb(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cth*cpsi,  sphi*sth*cpsi - cphi*spsi,  cphi*sth*cpsi + sphi*spsi],
        [cth*spsi,  sphi*sth*spsi + cphi*cpsi,  cphi*sth*spsi - sphi*cpsi],
        [   -sth,                   sphi*cth,                cphi*cth]
    ])

def plot3D(simData, numDataPoints, FPS, filename, figNo):
    # States (assumes NED; z positive Down)
    x = simData[:, 0]   # East
    y = simData[:, 1]   # North
    z = simData[:, 2]   # Down
    phi   = simData[:, 3]
    theta = simData[:, 4]
    psi   = simData[:, 5]

    # Downsample robustly
    Npts = min(numDataPoints, len(x))
    idx  = np.linspace(0, len(x) - 1, Npts).astype(int)

    E = x[idx]
    N = y[idx]
    D = z[idx]          # if your z was Up-positive, use D = -z[idx]
    ph = phi[idx]; th = theta[idx]; ps = psi[idx]

    # Figure & axes
    fig = plt.figure(figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue)
    ax  = fig.add_subplot(111, projection='3d')

    # Trajectory (we update its "tail" during animation for a sweeping effect)
    line, = ax.plot([], [], [], lw=1.8, color='k')

    # Arrow (triad) settings
    axis_len = 0.6  # [m]
    colors = dict(x='tab:red', y='tab:green', z='tab:blue')

    # Initialize quivers (None for now; we’ll create them on first frame)
    xb_quiv = yb_quiv = zb_quiv = None

    # Axes limits & labels
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D Trajectory plot')

    ax.invert_zaxis()
    ax.invert_xaxis()

    '''
    # Ground plane
    xg = np.linspace(-5, 5, 21)
    yg = np.linspace(-5, 5, 21)
    xx, yy = np.meshgrid(xg, yg)
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.25, linewidth=0, antialiased=False)
    '''
    
    # Legend proxies with LaTeX labels
    import matplotlib.lines as mlines

    h_x = mlines.Line2D([], [], color=colors['x'], linewidth=3, label=r'$x_b$')
    h_y = mlines.Line2D([], [], color=colors['y'], linewidth=3, label=r'$y_b$')
    h_z = mlines.Line2D([], [], color=colors['z'], linewidth=3, label=r'$z_b$')

    ax.legend(handles=[h_x, h_y, h_z], loc='upper center', ncol=3)

    def anim_function(num):
        nonlocal xb_quiv, yb_quiv, zb_quiv

        # tip index (>=1 so we have at least one point for the tail)
        i = max(1, num) - 1

        # Update swept trajectory
        line.set_data(E[:i], N[:i])
        line.set_3d_properties(D[:i])

        # Orientation at tip
        Cnb = euler2Cnb(ph[i], th[i], ps[i])
        xb  = Cnb[:, 0]   # x_b in NED
        yb  = Cnb[:, 1]   # y_b in NED
        zb  = Cnb[:, 2]   # z_b in NED

        # Tip position in NED
        Ex, Ny, Dz = E[i], N[i], D[i]

        # Remove previous arrows (cheap; only 3 objects)
        if xb_quiv is not None: xb_quiv.remove()
        if yb_quiv is not None: yb_quiv.remove()
        if zb_quiv is not None: zb_quiv.remove()

        # Draw new arrows with arrowheads using quiver
        xb_quiv = ax.quiver(Ex, Ny, Dz, xb[0], xb[1], xb[2],
                            length=axis_len, normalize=True, color=colors['x'])
        yb_quiv = ax.quiver(Ex, Ny, Dz, yb[0], yb[1], yb[2],
                            length=axis_len, normalize=True, color=colors['y'])
        zb_quiv = ax.quiver(Ex, Ny, Dz, zb[0], zb[1], zb[2],
                            length=axis_len, normalize=True, color=colors['z'])

        # Nice view
        ax.view_init(elev=10.0, azim=-120.0)
        return line, xb_quiv, yb_quiv, zb_quiv

    ani = animation.FuncAnimation(
        fig, anim_function,
        frames=Npts, interval=200, blit=False, repeat=True
    )

    # Save GIF (comment this if you just want an interactive window)
    ani.save(filename, writer=animation.PillowWriter(fps=FPS))




