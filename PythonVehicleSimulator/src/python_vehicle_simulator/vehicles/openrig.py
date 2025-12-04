#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def euler2Cnb(phi, theta, psi):
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
    OpenRig — thruster-controlled, with full 3D lever arms.

    Control behaviors (selected via constructor args or setters):
      - step mode: classic thruster step via allocator after step_time
      - free mode: thrusters off, initial φ/θ and p/q set once → natural decay

    Environment:
      - manually switchable horizontal current that affects damping via relative
        water velocity (nu_rel = nu - v_current_body).
    """

    def __init__(self,
                 mode="free",             # "step" or "free"
                 dof=5,                   # step DOF: 0..5 (X,Y,Z,roll,pitch,yaw)
                 step_amplitude=5.5,      # N (0..2) or Nm (3..5)
                 step_time=2.0,           # s
                 decay_axis="pitch",      # "roll" or "pitch"
                 decay_angle_deg=8.0,     # deg
                 decay_rate0=0.0):        # rad/s

        # ---- Pattern allocator (table-style signs) ----
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

        # --- UI / identity ---
        self.name = "OpenRig (thruster-controlled)"
        self.L = 0.60
        self.controls = ["T1","T2","T3","T4","T5","T6","T7","T8"]   # 8 thrusters
        self.dimU = len(self.controls)
        self.u_actual = np.zeros(self.dimU)

        # --- Step config ---
        if dof not in (0,1,2,3,4,5):
            raise ValueError("dof must be in 0..5")
        self.dof = int(dof)
        self.step_amplitude = float(step_amplitude)
        self.step_time = float(step_time)

        # --- Free-decay configuration ---
        decay_axis = str(decay_axis).lower()
        if decay_axis not in ("roll","pitch"):
            raise ValueError("decay_axis must be 'roll' or 'pitch'")
        self.free_decay_axis = decay_axis
        self.free_decay_angle_rad = float(np.deg2rad(decay_angle_deg))
        self.free_decay_rate0 = float(decay_rate0)
        self._free_ic_done = False  # apply IC once at t≈0

        # --- IMPORTANT: Fossen simulate() calls getattr(vehicle, controlMode)(t) ---
        # Always expose a callable named "stepInput":
        self.controlMode = "stepInput"

        # Internal behavior flag used INSIDE stepInput:
        self.mode = "free" if str(mode).lower() == "free" else "step"

        # Human-readable description
        if self.mode == "free":
            self.controlDescription = (
                f"Free decay in {decay_axis} from {decay_angle_deg:.1f}° "
                f"(rate0={decay_rate0:.3f} rad/s)"
            )
        else:
            self.controlDescription = (
                f"Thruster step in DOF {self.dof} after {self.step_time:.2f}s "
                f"(amp={self.step_amplitude:.2f})"
            )

        # --- Thruster geometry and limits ---
        self.T_MAX = 5.5           # [N] per thruster, symmetric ±

        # Vertical thrusters (rectangle), +Z thrust (positions relative to CO)
        self.rx_pitch = 0.1645     # [m] lever arm in x (pitch)
        self.ry_roll  = 0.254      # [m] lever arm in y (roll)
        self.z_vt     = 0.061339   # [m] height of vertical thrusters (rel. to CO)

        # Horizontal thrusters (square), ψ defines direction in XY (positions relative to CO)
        self.L_ht  = 0.150         # [m] ± lever arm in x and y
        self.z_ht  = 0.124         # [m] height of horizontal thrusters (rel. to CO)

        # ψ angles (rad): ht1..ht4
        self.psi_ht = np.array([
            -0.785398163,
            -2.35619449,
             2.35619449,
             0.785398163,
        ], dtype=float)

        # --- Physical properties ---
        self.m = 18.0
        self.g_acc = 9.81
        self.W = self.m * self.g_acc
        self.Buoy = 176.58
        self.neutral = True

        # CoG / CoB relative to CO
        self.r_g = np.array([0.0, 0.0, 0.12])
        self.r_b = np.array([0.0, 0.0, 0.09465])

        # Inertia (kg·m²)      
        Ixx, Iyy, Izz = 0.1231, 0.1228, 0.2457
        self.Ib = np.diag([Ixx, Iyy, Izz])

        # Added mass (negative diagonal)
        Xdu, Ydv, Zdw = 34.232, 46.3475, 172.76925
        Kdp, Mdq, Ndr = 4.5648055, 6.5716855, 4.071775
        self.MA = -np.diag([Xdu, Ydv, Zdw, Kdp, Mdq, Ndr])

        # --- Rigid-body mass about CoM (no CoG coupling terms) ---
        rgx = rgy = rgz = 0.0
        rg_skew = np.array([[0, -rgz,  rgy],
                            [rgz,   0, -rgx],
                            [-rgy, rgx,  0]])
        self.MRB = np.zeros((6,6))
        self.MRB[:3,:3] = self.m * np.eye(3)
        self.MRB[:3,3:] = -self.m * rg_skew   # zero
        self.MRB[3:,:3] =  self.m * rg_skew   # zero
        self.MRB[3:,3:] = self.Ib

        # Linear damping (Ns/m, Nms/rad)
        Xu, Yv, Zw = 27.432, 34.234, 86.353 #138.1305
        Kp, Mq, Nr = 2.1186255, 4.0203245, 6.16035
        self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])

        # Small-angle hydrostatic stiffness (Nm/rad)
        self.K_phi   = 4.34905
        self.K_theta = 4.63692

        # Total mass matrix (about CoM)
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # States
        self.nu  = np.zeros(6)   # [u, v, w, p, q, r]
        self.eta = np.zeros(6)   # [x, y, z, φ, θ, ψ]

        # --- Environmental current (manually switchable) ---
        # current_frame: "earth" => direction is earth azimuth (x=N, y=E)
        #                 "body"  => direction fixed in body (x=surge, y=sway)
        self.current_on = False
        self.current_speed = 0.0          # [m/s]
        self.current_dir_deg = 90.0       # [deg] 0 = +X, 90 = +Y
        self.current_frame = "earth"      # or "body"

        # Allocation
        self.rebuild_B()

        # >>> APPLY IC IMMEDIATELY IF FREE-DECAY MODE <<<
        if self.mode == "free":
            self._apply_free_decay_ic()

        # Allocation
        self.rebuild_B()

    # ----------------------- Mode helpers -----------------------

    def set_step_response(self, dof: int = None, amplitude: float = None, step_time: float = None):
        """Switch to classic step-input mode and update parameters."""
        if dof is not None:
            if dof not in (0,1,2,3,4,5):
                raise ValueError("dof must be in 0..5")
            self.dof = int(dof)
        if amplitude is not None:
            self.step_amplitude = float(amplitude)
        if step_time is not None:
            self.step_time = float(step_time)
        self.mode = "step"
        self.controlMode = "stepInput"  # keep callable name stable
        self.controlDescription = (
            f"Thruster step in DOF {self.dof} after {self.step_time:.2f}s (amp={self.step_amplitude:.2f})"
        )

    # --- apply free-decay initial condition immediately ---
    def _apply_free_decay_ic(self):
        """Set the initial attitude/rate for free-decay and mark as applied."""
        if self.free_decay_axis == "roll":
            self.eta[3] = self.free_decay_angle_rad   # φ(0)
            self.nu[3]  = self.free_decay_rate0       # p(0)
        else:  # pitch
            self.eta[4] = self.free_decay_angle_rad   # θ(0)
            self.nu[4]  = self.free_decay_rate0       # q(0)
        self._free_ic_done = True

    def set_free_decay(self, axis="pitch", angle_deg=8.0, rate0=0.0):
        """Switch to free-decay mode and update parameters."""
        axis = axis.lower()
        if axis not in ("roll","pitch"):
            raise ValueError("axis must be 'roll' or 'pitch'")
        self.free_decay_axis = axis
        self.free_decay_angle_rad = float(np.deg2rad(angle_deg))
        self.free_decay_rate0 = float(rate0)
        self.mode = "free"
        self.controlMode = "stepInput"  # keep callable name stable
        self.controlDescription = f"Free decay in {axis} from {angle_deg:.1f}° (rate0={rate0:.3f} rad/s)"
        self._free_ic_done = False

    # ----------------------- Current helpers -----------------------
    def set_side_current(self, on: bool = True, speed: float = 0.2,
                         direction_deg: float = 90.0, frame: str = "earth"):
        """
        Enable/disable a horizontal current.
        - speed: magnitude in m/s
        - direction_deg:
            * frame == "earth": azimuth in earth frame (x=N, y=E), 0°=+x, 90°=+y
            * frame == "body" : azimuth in body frame (x=surge, y=sway)
        - frame: "earth" or "body"
        """
        frame = frame.lower()
        if frame not in ("earth", "body"):
            raise ValueError("frame must be 'earth' or 'body'")
        self.current_on = bool(on)
        self.current_speed = float(speed)
        self.current_dir_deg = float(direction_deg)
        self.current_frame = frame

    def _current_linear_body(self) -> np.ndarray:
        """
        Current linear velocity [u_c, v_c, 0] expressed in the BODY frame.
        If frame == 'body', use the body azimuth directly.
        If frame == 'earth', rotate earth current into body with C_bn = R_nb^T.
        """
        if not self.current_on or self.current_speed <= 0.0:
            return np.zeros(3)

        az = np.deg2rad(self.current_dir_deg)
        if self.current_frame == "body":
            # body-fixed azimuth (surge/sway plane)
            v_b = np.array([np.cos(az), np.sin(az), 0.0]) * self.current_speed
            return v_b

        # earth-fixed azimuth (x=N, y=E)
        v_e = np.array([np.cos(az), np.sin(az), 0.0]) * self.current_speed

        # Rotate earth vector into body: C_bn = R_nb^T
        phi, theta, psi = self.eta[3:6]
        Rnb = euler2Cnb(phi, theta, psi)
        Cbn = Rnb.T
        return Cbn @ v_e

    # ----------------------- Geometry → B -----------------------
    def allocate_by_pattern(self, dof, tau_d_scalar):
        """
        STRICT pattern allocator using sign matrix S:
        Returns T ∈ R^8 in the order [vt1,vt2,vt3,vt4, ht1,ht2,ht3,ht4].
        """
        s = self.S[dof].astype(float)  # {-1,0,+1}
        if not np.any(s):
            return np.zeros(8)
        T = np.zeros(8, dtype=float)
        direction = 1.0 if tau_d_scalar >= 0.0 else -1.0
        T[0:8] = direction * self.T_MAX * s[0:8]
        return T

    def rebuild_B(self):
        """Build allocation matrix B (6x8) with m = (r - r_g) × f moments about CoM."""
        B = np.zeros((6, 8))

        # --- Vertical thrusters: +Z ---
        vt_pos_CO = np.array([
            [ +self.rx_pitch, +self.ry_roll,  self.z_vt],  # vt1
            [ -self.rx_pitch, +self.ry_roll,  self.z_vt],  # vt2
            [ -self.rx_pitch, -self.ry_roll,  self.z_vt],  # vt3
            [ +self.rx_pitch, -self.ry_roll,  self.z_vt],  # vt4
        ], dtype=float)
        f_vt = np.array([0.0, 0.0, 1.0], dtype=float)

        for i in range(4):
            r = vt_pos_CO[i] - self.r_g
            m = np.cross(r, f_vt)
            B[:, i] = np.hstack((f_vt, m))

        # --- Horizontal thrusters: dir from ψ, positions relative to CO ---
        L = self.L_ht
        ht_pos_CO = np.array([
            [ +L, +L, self.z_ht],
            [ -L, +L, self.z_ht],
            [ -L, -L, self.z_ht],
            [ +L, -L, self.z_ht],
        ], dtype=float)

        for j in range(4):
            psi = self.psi_ht[j]
            f = np.array([np.cos(psi), np.sin(psi), 0.0], dtype=float)  # Rz(ψ)·e_x
            r = ht_pos_CO[j] - self.r_g
            m = np.cross(r, f)
            B[:, 4 + j] = np.hstack((f, m))

        self.B = B

    def set_thruster_heights(self, z_vt=None, z_ht=None):
        """Change heights (relative to CO) and rebuild B."""
        if z_vt is not None:
            self.z_vt = float(z_vt)
        if z_ht is not None:
            self.z_ht = float(z_ht)
        self.rebuild_B()

    # ---------------------- Restoring & C -----------------------
    def restoring(self, eta):
        """
        Neutral buoyancy (W==B) → restoring moment due to CoM–CoB offset:
            g = [ 0 ;
                -W * (r_bg x b3) ],  r_bg = r_b - r_g, b3 = body z-axis in NED.
        """
        phi, theta, psi = eta[3:6]
        sphi, cphi = np.sin(phi),  np.cos(phi)
        sth,  cth  = np.sin(theta), np.cos(theta)
        b3 = np.array([-sth, sphi*cth, cphi*cth], dtype=float)
        r_bg = (self.r_b - self.r_g).astype(float)
        W = float(self.W)

        g = np.zeros(6, dtype=float)
        g[3:] = -W * np.cross(r_bg, b3)
        return g

    def coriolis(self, nu):
        # Simple RB coriolis based on MRB about CoM (no CoG coupling)
        m = self.MRB[0,0]
        u, v, w, p, q, r = nu
        C = np.zeros((6,6))
        C[0,4] =  m*w;  C[0,5] = -m*v
        C[1,3] = -m*w;  C[1,5] =  m*u
        C[2,3] =  m*v;  C[2,4] = -m*u
        return C

    # ----------------------- Simulator hooks --------------------
    def stepInput(self, t):
        """
        Called by simulate() via getattr(vehicle, vehicle.controlMode)(t).
        We keep controlMode == "stepInput" and switch behavior using self.mode.
        """
        if self.mode == "free":
            # Apply initial condition once at t≈0, thrusters off
            if not self._free_ic_done and t <= 1e-12:
                if self.free_decay_axis == "roll":
                    self.eta[3] = self.free_decay_angle_rad   # φ
                    self.nu[3]  = self.free_decay_rate0       # p
                else:  # pitch
                    self.eta[4] = self.free_decay_angle_rad   # θ
                    self.nu[4]  = self.free_decay_rate0       # q
                self._free_ic_done = True
            return np.zeros(8, dtype=float)

        # step mode
        if t <= self.step_time:
            return np.zeros(8, dtype=float)
        return self.allocate_by_pattern(self.dof, self.step_amplitude)

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        Integrate one step with forces/moments from thrusters + hydrostatics + damping + Coriolis.
        Environmental current enters via relative water velocity in the damping term.
        As an extra guard, enforce thrusters off when in free-decay mode.
        """
        if self.mode == "free":
            T = np.zeros(8, dtype=float)
        else:
            T = np.array(u_control, dtype=float).reshape(8,)

        # Thruster-induced body wrench
        tau_thrusters = self.B @ T
        self.u_actual = T.copy()      # for plotControls

        # Restoring and Coriolis
        g_eta = self.restoring(self.eta)
        C = self.coriolis(self.nu)

        # --- Relative velocity vs. water (only linear DOFs) ---
        v_c_body = self._current_linear_body()     # [u_c, v_c, 0]
        nu_rel = self.nu.copy()
        nu_rel[:3] = self.nu[:3] - v_c_body        # translational component is relative

        # Damping acts on relative water velocity
        tau_damp = self.D @ nu_rel

        # Total external body wrench (excluding Coriolis term which is state-coupling)
        # tau_total = tau_thrusters - tau_damp - g_eta
        # Final equation: M * nu_dot = tau_thrusters - (C @ nu + tau_damp) - g_eta
        nu_dot = self.Minv @ (tau_thrusters - (C @ self.nu + tau_damp) - g_eta)
        self.nu += nu_dot * sampleTime

        # Kinematics
        Rnb = euler2Cnb(self.eta[3], self.eta[4], self.eta[5])
        vel_earth = np.hstack((Rnb @ self.nu[:3], self.nu[3:6]))
        self.eta += vel_earth * sampleTime
        return self.nu, self.u_actual


# ----------------------- Minimal usage example -----------------------
if __name__ == "__main__":
    rig = OpenRig(mode="step", dof=1, step_amplitude=5.5, step_time=2.0)

    # Turn ON a side current: 0.25 m/s from +Y (east) in earth frame
    rig.set_side_current(on=True, speed=0.25, direction_deg=90.0, frame="earth")

    # Example single dynamics step (replace with your simulate() loop)
    dt = 0.01
    u_control = rig.allocate_by_pattern(rig.dof, rig.step_amplitude)
    for k in range(1000):
        t = k * dt
        u_cmd = rig.stepInput(t)
        rig.dynamics(rig.eta, rig.nu, rig.u_actual, u_cmd, dt)

    print("Final state eta:", rig.eta)
    print("Final body rates nu:", rig.nu)
