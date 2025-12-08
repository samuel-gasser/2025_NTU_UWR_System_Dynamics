#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def euler2Cnb(phi, theta, psi):
    """Rotation BODY -> NED (Fossen R_nb)."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cth*cpsi,  sphi*sth*cpsi - cphi*spsi,  cphi*sth*cpsi + sphi*spsi],
        [cth*spsi,  sphi*sth*spsi + cphi*cpsi,  cphi*sth*spsi - sphi*cpsi],
        [   -sth,                   sphi*cth,                cphi*cth]
    ])


class OpenRig:
    """
    OpenRig — AUV with simple position-hold PI controller against a current.

    Modes:
      - mode="hold":
          * 0 .. controller_start_time: thrusters off, system drifts with current
          * t >= controller_start_time: position-hold (x,y,psi) at origin
      - mode="step":
          * 0 .. step_time: thrusters off
          * step_time .. controller_start_time: classic DOF-step (pattern)
          * t >= controller_start_time: position-hold (x,y,psi)
      - mode="free":
          * free decay in roll/pitch, thrusters always off
    """

    def __init__(self,
                 mode="hold",
                 dof=0,                  # only for mode="step"
                 step_amplitude=5.5,
                 step_time=2.0,
                 decay_axis="pitch",
                 decay_angle_deg=8.0,
                 decay_rate0=0.0):

        # ---- Pattern allocator (for step-mode) ----
        # Rows = [X, Y, Z, K, M, N], Cols = [vt1, vt2, vt3, vt4, ht1, ht2, ht3, ht4]
        self.S = np.array([
            [ 0,  0,  0,  0,  -1, +1, +1, -1],   # X (surge): HT only
            [ 0,  0,  0,  0,  +1, +1, -1, -1],   # Y (sway):  HT only
            [ -1, -1, -1, -1,  0,  0,  0,  0],   # Z (heave): VT only
            [ -1, -1, +1, +1,  0,  0,  0,  0],   # K (roll):  VT
            [ +1, -1, -1, +1,  0,  0,  0,  0],   # M (pitch): VT
            [ 0,  0,  0,  0,  +1, -1, +1, -1],   # N (yaw):   HT only
        ], dtype=float)

        self.use_pattern_allocator = True

        # --- Identity / UI ---
        self.name = "OpenRig (position-hold PI)"
        self.L = 0.60
        self.controls = ["T1","T2","T3","T4","T5","T6","T7","T8"]
        self.dimU = len(self.controls)
        self.u_actual = np.zeros(self.dimU)

        # --- Step-config (only for mode='step') ---
        if dof not in (0,1,2,3,4,5):
            raise ValueError("dof must be in 0..5")
        self.dof = int(dof)
        self.step_amplitude = float(step_amplitude)
        self.step_time = float(step_time)

        # --- Free-decay config ---
        decay_axis = str(decay_axis).lower()
        if decay_axis not in ("roll", "pitch"):
            raise ValueError("decay_axis must be 'roll' or 'pitch'")
        self.free_decay_axis = decay_axis
        self.free_decay_angle_rad = float(np.deg2rad(decay_angle_deg))
        self.free_decay_rate0 = float(decay_rate0)
        self._free_ic_done = False

        # --- simulate()-hook ---
        self.controlMode = "stepInput"

        mode_str = str(mode).lower()
        if mode_str in ("hold", "dp", "position", "poshold"):
            self.mode = "hold"
        elif mode_str == "free":
            self.mode = "free"
        else:
            self.mode = "step"

        if self.mode == "free":
            self.controlDescription = (
                f"Free decay in {decay_axis} from {decay_angle_deg:.1f}° "
                f"(rate0={decay_rate0:.3f} rad/s)"
            )
        elif self.mode == "step":
            self.controlDescription = (
                f"Step in DOF {self.dof} (amp={self.step_amplitude:.2f}) from t={self.step_time:.2f}s, "
                f"then position-hold"
            )
        else:
            self.controlDescription = "Position-hold controller (x,y,psi) at origin"

        # --- Position-hold settings ---
        self.controller_start_time = 5.0  # [s] – controller turns on here

        # Position reference (origin)
        self.x_ref = 0.0
        self.y_ref = 0.0

        # Yaw reference: freeze initial yaw at first controller call
        self.psi_ref = 0.0
        self._poshold_initialized = False

        # PI gains in BODY for position
        self.pos_Kp = np.array([500.0, 500.0])   # [N/m] for X,Y
        self.pos_Ki = np.array([ 10.0,  10.0])   # [N/(m·s)] for X,Y

        # Integral states for position error in BODY
        self.pos_int = np.zeros(2)            # [int_ex_b, int_ey_b]
        self._pi_last_t = None                # last time stamp for dt

        # Yaw PD gains (keep PD here for stability)
        self.yaw_Kp = 3.0     # [Nm/rad]
        self.yaw_Kd = 2.0     # [Nm/(rad/s)]

        # Deadbands and wrench limits
        self.pos_deadband = 0.05              # [m] around origin in BODY
        self.yaw_deadband = np.deg2rad(2.0)   # [rad] ≈ 2°
        self.vel_deadband = 0.01              # [m/s] for u,v

        self.tau_X_max = 200.0   # [N]
        self.tau_Y_max = 200.0   # [N]
        self.tau_N_max = 100.0   # [Nm]

        # --- Thruster geometry ---
        self.T_MAX = 10

        self.rx_pitch = 0.1645
        self.ry_roll  = 0.254
        self.z_vt     = 0.061339

        self.L_ht  = 0.150
        self.z_ht  = 0.124

        self.psi_ht = np.array([
            -0.785398163,
            -2.35619449,
             2.35619449,
             0.785398163,
        ], dtype=float)

        # --- Physical properties (from your original model) ---
        self.m = 18.0
        self.g_acc = 9.81
        self.W = self.m * self.g_acc
        self.Buoy = 176.58
        self.neutral = True

        self.r_g = np.array([0.0, 0.0, 0.12])
        self.r_b = np.array([0.0, 0.0, 0.09465])

        Ixx, Iyy, Izz = 0.1231, 0.1228, 0.2457
        self.Ib = np.diag([Ixx, Iyy, Izz])

        Xdu, Ydv, Zdw = 34.232, 46.3475, 172.76925
        Kdp, Mdq, Ndr = 4.5648055, 6.5716855, 4.071775
        self.MA = -np.diag([Xdu, Ydv, Zdw, Kdp, Mdq, Ndr])

        rgx = rgy = rgz = 0.0
        rg_skew = np.array([[0, -rgz,  rgy],
                            [rgz,   0, -rgx],
                            [-rgy, rgx,  0]])
        self.MRB = np.zeros((6,6))
        self.MRB[:3,:3] = self.m * np.eye(3)
        self.MRB[:3,3:] = -self.m * rg_skew
        self.MRB[3:,:3] =  self.m * rg_skew
        self.MRB[3:,3:] = self.Ib

        Xu, Yv, Zw = 27.432, 34.234, 86.353
        Kp, Mq, Nr = 2.1186255, 4.0203245, 6.16035
        self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])

        self.K_phi   = 4.34905
        self.K_theta = 4.63692

        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # States
        self.nu  = np.zeros(6)   # [u, v, w, p, q, r]
        self.eta = np.zeros(6)   # [x, y, z, φ, θ, ψ]

        # Initial psi_ref (will be frozen at first controller call)
        self.psi_ref = self.eta[5]

        # Current
        self.current_on = False
        self.current_speed = 0.0
        self.current_dir_deg = 90.0
        self.current_frame = "earth"

        # Allocation matrix
        self.rebuild_B()

        if self.mode == "free":
            self._apply_free_decay_ic()

        self.rebuild_B()

    # ----------------------- Current helpers -----------------------

    def set_side_current(self, on: bool = True, speed: float = 0.2,
                         direction_deg: float = 90.0, frame: str = "earth"):
        frame = frame.lower()
        if frame not in ("earth", "body"):
            raise ValueError("frame must be 'earth' or 'body'")
        self.current_on = bool(on)
        self.current_speed = float(speed)
        self.current_dir_deg = float(direction_deg)
        self.current_frame = frame

    def _current_linear_body(self) -> np.ndarray:
        if not self.current_on or self.current_speed <= 0.0:
            return np.zeros(3)

        az = np.deg2rad(self.current_dir_deg)
        if self.current_frame == "body":
            return np.array([np.cos(az), np.sin(az), 0.0]) * self.current_speed

        v_e = np.array([np.cos(az), np.sin(az), 0.0]) * self.current_speed

        phi, theta, psi = self.eta[3:6]
        Rnb = euler2Cnb(phi, theta, psi)
        Cbn = Rnb.T
        return Cbn @ v_e

    # ----------------------- Geometry → B -----------------------

    def rebuild_B(self):
        B = np.zeros((6, 8))

        vt_pos_CO = np.array([
            [ +self.rx_pitch, +self.ry_roll,  self.z_vt],
            [ -self.rx_pitch, +self.ry_roll,  self.z_vt],
            [ -self.rx_pitch, -self.ry_roll,  self.z_vt],
            [ +self.rx_pitch, -self.ry_roll,  self.z_vt],
        ], dtype=float)
        f_vt = np.array([0.0, 0.0, 1.0], dtype=float)

        for i in range(4):
            r = vt_pos_CO[i] - self.r_g
            m = np.cross(r, f_vt)
            B[:, i] = np.hstack((f_vt, m))

        L = self.L_ht
        ht_pos_CO = np.array([
            [ +L, +L, self.z_ht],
            [ -L, +L, self.z_ht],
            [ -L, -L, self.z_ht],
            [ +L, -L, self.z_ht],
        ], dtype=float)

        for j in range(4):
            psi = self.psi_ht[j]
            f = np.array([np.cos(psi), np.sin(psi), 0.0], dtype=float)
            r = ht_pos_CO[j] - self.r_g
            m = np.cross(r, f)
            B[:, 4 + j] = np.hstack((f, m))

        self.B = B

    def allocate_by_pattern(self, dof, tau_d_scalar):
        s = self.S[dof].astype(float)
        if not np.any(s):
            return np.zeros(8)
        T = np.zeros(8, dtype=float)
        direction = 1.0 if tau_d_scalar >= 0.0 else -1.0
        T[0:8] = direction * self.T_MAX * s[0:8]
        return T

    # ---------------------- Restoring & C -----------------------

    def restoring(self, eta):
        phi, theta, psi = eta[3:6]
        sphi, cphi = np.sin(phi), np.cos(phi)
        sth,  cth  = np.sin(theta), np.cos(theta)
        b3 = np.array([-sth, sphi*cth, cphi*cth], dtype=float)
        r_bg = (self.r_b - self.r_g).astype(float)
        W = float(self.W)

        g = np.zeros(6, dtype=float)
        g[3:] = -W * np.cross(r_bg, b3)
        return g

    def coriolis(self, nu):
        m = self.MRB[0,0]
        u, v, w, p, q, r = nu
        C = np.zeros((6,6))
        C[0,4] =  m*w;  C[0,5] = -m*v
        C[1,3] = -m*w;  C[1,5] =  m*u
        C[2,3] =  m*v;  C[2,4] = -m*u
        return C

    # -------------------- Free-decay helper --------------------

    def _apply_free_decay_ic(self):
        if self.free_decay_axis == "roll":
            self.eta[3] = self.free_decay_angle_rad
            self.nu[3]  = self.free_decay_rate0
        else:
            self.eta[4] = self.free_decay_angle_rad
            self.nu[4]  = self.free_decay_rate0
        self._free_ic_done = True

    # -------------------- Position-hold PI controller --------------------

    def _position_hold_control(self, t: float) -> np.ndarray:
        """
        Position-hold:
          - Compute error in NED (ref - actual), then rotate to BODY.
          - PI controller on BODY position (X,Y).
          - PD on yaw (psi).
          - Deadband around the origin.
          - Wrench saturation.
          - Allocation using only horizontal thrusters (T5..T8).
        """
        # Freeze psi_ref at first call
        if not self._poshold_initialized:
            self.psi_ref = self.eta[5]
            self._poshold_initialized = True
            self._pi_last_t = t

        x, y, _, phi, theta, psi = self.eta
        u, v, w, p, q, r = self.nu

        # Rotation matrices
        Rnb = euler2Cnb(phi, theta, psi)   # BODY -> NED
        Cbn = Rnb.T                        # NED -> BODY

        # Position error in NED: ref - actual
        e_n = np.array([self.x_ref - x, self.y_ref - y, 0.0])

        # Position error in BODY
        e_b = Cbn @ e_n
        e_bx, e_by, _ = e_b

        # Yaw error (ref - actual)
        e_psi = self.psi_ref - psi
        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))

        # Deadband around zero to avoid chattering
        if (abs(e_bx) < self.pos_deadband and
            abs(e_by) < self.pos_deadband and
            abs(e_psi) < self.yaw_deadband and
            np.hypot(u, v) < self.vel_deadband):
            return np.zeros(8, dtype=float)

        # --- dt for PI integration ---
        dt = t - self._pi_last_t if self._pi_last_t is not None else 0.0
        self._pi_last_t = t

        # Integrate position errors in BODY (simple anti-windup later)
        self.pos_int += np.array([-e_bx, -e_by]) * dt

        # Anti-windup: bound integral so that Ki * int <= tau_max
        int_lim_x = self.tau_X_max / max(self.pos_Ki[0], 1e-6)
        int_lim_y = self.tau_Y_max / max(self.pos_Ki[1], 1e-6)
        self.pos_int[0] = np.clip(self.pos_int[0], -int_lim_x, int_lim_x)
        self.pos_int[1] = np.clip(self.pos_int[1], -int_lim_y, int_lim_y)

        # PI in BODY for X,Y
        tau_X = self.pos_Kp[0] * -e_bx + self.pos_Ki[0] * self.pos_int[0]
        tau_Y = self.pos_Kp[1] * -e_by + self.pos_Ki[1] * self.pos_int[1]

        # PD on yaw
        tau_N = self.yaw_Kp * -e_psi - self.yaw_Kd * r

        # Wrench saturation
        tau_X = np.clip(tau_X, -self.tau_X_max, self.tau_X_max)
        tau_Y = np.clip(tau_Y, -self.tau_Y_max, self.tau_Y_max)
        tau_N = np.clip(tau_N, -self.tau_N_max, self.tau_N_max)

        tau_xyN = np.array([tau_X, tau_Y, tau_N], dtype=float)

        # B_sub: only horizontal thrusters, rows X,Y,N
        B_sub = self.B[[0, 1, 5], 4:8]   # 3x4

        # Least-squares allocation
        T_ht, *_ = np.linalg.lstsq(B_sub, tau_xyN, rcond=None)

        # Thruster saturation
        T_ht = np.clip(T_ht, -self.T_MAX, self.T_MAX)

        # Full thruster vector (verticals off)
        T = np.zeros(8, dtype=float)
        T[4:8] = T_ht
        return T

    # ----------------------- Simulator hook --------------------

    def stepInput(self, t):
        """
        simulate() calls getattr(vehicle, controlMode)(t).

        FREE:
          - thrusters off, only roll/pitch IC

        HOLD:
          - 0 .. controller_start_time: thrusters off (drift with current)
          - t >= controller_start_time: position-hold controller

        STEP:
          - 0 .. step_time: thrusters off
          - step_time .. controller_start_time: classic step
          - t >= controller_start_time: position-hold controller
        """
        if self.mode == "free":
            if not self._free_ic_done and t <= 1e-12:
                self._apply_free_decay_ic()
            return np.zeros(8, dtype=float)

        if self.mode == "hold":
            if t < self.controller_start_time:
                return np.zeros(8, dtype=float)
            return self._position_hold_control(t)

        # mode == "step"
        t_ctrl = self.controller_start_time
        if t >= t_ctrl:
            return self._position_hold_control(t)

        if t <= self.step_time:
            return np.zeros(8, dtype=float)
        return self.allocate_by_pattern(self.dof, self.step_amplitude)

    # ----------------------- Dynamics -----------------------

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        One integration step of 6-DOF motion.
        """
        if self.mode == "free":
            T = np.zeros(8, dtype=float)
        else:
            T = np.array(u_control, dtype=float).reshape(8,)

        tau_thrusters = self.B @ T
        self.u_actual = T.copy()

        g_eta = self.restoring(self.eta)
        C = self.coriolis(self.nu)

        v_c_body = self._current_linear_body()
        nu_rel = self.nu.copy()
        nu_rel[:3] = self.nu[:3] - v_c_body

        tau_damp = self.D @ nu_rel

        nu_dot = self.Minv @ (tau_thrusters - (C @ self.nu + tau_damp) - g_eta)
        self.nu += nu_dot * sampleTime

        Rnb = euler2Cnb(self.eta[3], self.eta[4], self.eta[5])
        vel_earth = np.hstack((Rnb @ self.nu[:3], self.nu[3:6]))
        self.eta += vel_earth * sampleTime
        return self.nu, self.u_actual


# ----------------------- Minimal test with surge plot -----------------------
if __name__ == "__main__":
    rig = OpenRig(mode="hold")

    # Current: 0.25 m/s from +Y in earth frame
    rig.set_side_current(on=True, speed=0.25, direction_deg=90.0, frame="earth")

    dt = 0.02
    T_end = 60.0
    N = int(T_end / dt)

    t_hist = np.zeros(N)
    x_hist = np.zeros(N)   # surge position x (NED)
    u_hist = np.zeros(N)   # surge velocity u (BODY)

    for k in range(N):
        t = k * dt
        u_cmd = rig.stepInput(t)
        rig.dynamics(rig.eta, rig.nu, rig.u_actual, u_cmd, dt)

        t_hist[k] = t
        x_hist[k] = rig.eta[0]   # x-position in NED
        u_hist[k] = rig.nu[0]    # surge velocity u in BODY

    # ---- Plot: x(t) and u(t) with two y-axes ----
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Left axis: surge position x(t)
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("surge position x [m]", color="tab:blue")
    ax1.plot(t_hist, x_hist, color="tab:blue", label="x(t)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)

    # Right axis: surge velocity u(t)
    ax2 = ax1.twinx()
    ax2.set_ylabel("surge velocity u [m/s]", color="tab:orange")
    ax2.plot(t_hist, u_hist, color="tab:orange", linestyle="--", label="u(t)")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Surge position x(t) and surge velocity u(t)")
    plt.tight_layout()

    # Save as PNG in the current working directory
    outfile = "surge_x_u.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Plot saved as '{outfile}'")

    # Optional: only use show() if your environment supports it
    # plt.show()

