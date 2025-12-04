#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py: Main program for the Python Vehicle Simulator, which can be used
    to simulate and test guidance, navigation and control (GNC) systems.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd edition, John Wiley & Sons, Chichester, UK. 
URL: https://www.fossen.biz/wiley  
    
Author:     Thor I. Fossen
"""
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys 
import webbrowser
import matplotlib.pyplot as plt
from python_vehicle_simulator.vehicles import (
    DSRV, frigate, otter, ROVzefakkel, semisub, shipClarke83, supply, tanker, 
    remus100, torpedo, OpenRig 
)
from python_vehicle_simulator.lib import (
    printSimInfo, printVehicleinfo, simulate, plotVehicleStates, plotControls, 
    plot3D
)

### Simulation parameters ###
sampleTime = 0.02                   # sample time [seconds]
N = 1000                            # number of samples

# 3D plot and animation settings where browser = {firefox,chrome,safari,etc.}
numDataPoints = 50                  # number of 3D data points
FPS = 10                            # frames per second (animated GIF)
filename = '3D_animation.gif'       # data file for animated GIF
browser = 'safari'                  # browser for visualization of animated GIF

"""
Vehicle constructors:
  DSRV('depthAutopilot', z_d)                                        
  frigate('headingAutopilot', U, psi_d)
  otter('headingAutopilot', psi_d, V_c, beta_c, tau_X)                  
  ROVzefakkel('headingAutopilot', U, psi_d)                          
  semisub('DPcontrol', x_d, y_d, psi_d, V_c, beta_c)                      
  shipClarke83('headingAutopilot', psi_d, L, B, T, Cb, V_c, beta_c, tau_X)  
  supply('DPcontrol', x_d, y_d, psi_d, V_c, beta_c)      
  tanker('headingAutopilot', psi_d, V_c, beta_c, depth)    
  remus100('depthHeadingAutopilot', z_d, psi_d, V_c, beta_c)             
  torpedo('depthHeadingAutopilot', z_d, psi_d, V_c, beta_c)             

Call constructors without arguments to test step inputs, e.g. DSRV(), otter(), etc.
"""

def export_to_excel(simTime, simData, vehicle, out_dir=".", sampleTime=None, N=None):
    """
    Write simulation output to an Excel file:
      - Sheet 'states': time + simData columns
      - Sheet 'meta'  : basic run info (+ current settings if available)

    Layout compatibility:
      simData = [x, y, z, phi, theta, psi, u, v, w, p, q, r,
                 control_cmd..., control_act...]
    """
    out_dir = Path(r"C:\Users\samue\Documents\PythonVehicleSimulator\src\python_vehicle_simulator\exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    ncols = simData.shape[1] if simData.ndim == 2 else 1
    DOF = 6

    if hasattr(vehicle, "controls"):
        try:
            n_controls = len(vehicle.controls)
        except Exception:
            n_controls = 0
    else:
        n_controls = 0

    state_cols = ["x","y","z","phi","theta","psi","u","v","w","p","q","r"]

    if ncols == 2*DOF + 2*n_controls and n_controls > 0:
        cols = (
            state_cols +
            [f"{c}_cmd" for c in vehicle.controls] +
            [f"{c}_act" for c in vehicle.controls]
        )
    elif ncols == 2*DOF + n_controls and n_controls > 0:
        cols = state_cols + list(vehicle.controls)
    else:
        base20 = ["x","y","z","phi","theta","psi","u","v","w","p","q","r",
                  "vt1","vt2","vt3","vt4","ht1","ht2","ht3","ht4"]
        cols = base20[:ncols] if ncols <= 20 else base20 + [f"s{i}" for i in range(21, ncols + 1)]

    df_states = pd.DataFrame(simData, columns=cols)
    df_states.insert(0, "time", simTime)

    # Meta sheet
    meta_keys = ["vehicle", "sampleTime", "N", "exported_at"]
    meta_vals = [
        getattr(vehicle, "name", type(vehicle).__name__),
        sampleTime,
        N,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]

    # If OpenRig current settings exist, record them
    for attr in ["current_on", "current_speed", "current_dir_deg", "current_frame"]:
        if hasattr(vehicle, attr):
            meta_keys.append(attr)
            meta_vals.append(getattr(vehicle, attr))

    df_meta = pd.DataFrame({"key": meta_keys, "value": meta_vals})

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    vname = getattr(vehicle, "name", type(vehicle).__name__).replace(" ", "_")
    out_path = out_dir / f"results_{vname}_{stamp}.xlsx"
    if out_path.exists():
        out_path = out_dir / f"results_{vname}_{stamp}_{np.random.randint(1000):03d}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_states.to_excel(xw, index=False, sheet_name="states")
        df_meta.to_excel(xw, index=False, sheet_name="meta")

    print(f"[Excel] Wrote {out_path}")
    return out_path


# ---------- NEW: helper to configure side current for OpenRig ----------
def maybe_enable_current(vehicle):
    """
    Ask the user and (if OpenRig) enable/disable the side current.
    For other vehicle types, this does nothing.
    """
    if not isinstance(vehicle, OpenRig):
        return

    use = input("Enable side current? [y/N]: ").strip().lower()
    if use != "y":
        # Ensure it's off
        try:
            vehicle.set_side_current(on=False)
        except AttributeError:
            pass
        print("Side current: OFF")
        return

    # Read settings with safe fallbacks
    try:
        spd = float(input("Current speed [m/s] (e.g. 0.25): ").strip())
    except Exception:
        spd = 0.25

    try:
        deg = float(input("Direction azimuth [deg] (0=+X, 90=+Y): ").strip())
    except Exception:
        deg = 90.0

    frame = input("Frame 'earth' or 'body' [earth]: ").strip().lower() or "earth"
    if frame not in ("earth", "body"):
        frame = "earth"

    # Apply to vehicle
    try:
        vehicle.set_side_current(on=True, speed=spd, direction_deg=deg, frame=frame)
        print(f"Side current: ON (speed={spd} m/s, dir={deg} deg, frame={frame})")
    except AttributeError:
        # In case a different OpenRig without the method is imported
        print("Warning: vehicle does not support set_side_current(...). Ignoring.")


### Main program ###
def main():
    printSimInfo() 

    no = input("Please enter a vehicle no.: ").strip()  

    vehicleOptions = {
        '1': lambda: DSRV('depthAutopilot', 60.0),
        '2': lambda: frigate('headingAutopilot', 10.0, 100.0),
        '3': lambda: otter('headingAutopilot', 100.0, 0.3, -30.0, 200.0),
        '4': lambda: ROVzefakkel('headingAutopilot', 3.0, 100.0),
        '5': lambda: semisub('DPcontrol', 10.0, 10.0, 40.0, 0.5, 190.0),
        '6': lambda: shipClarke83('headingAutopilot', -20.0, 70, 8, 6, 0.7, 0.5, 10.0, 1e5),
        '7': lambda: supply('DPcontrol', 4.0, 4.0, 50.0, 0.5, 20.0),
        '8': lambda: tanker('headingAutopilot', -20, 0.5, 150, 20, 80),
        '9': lambda: remus100('depthHeadingAutopilot', 30, 50, 1525, 0.5, 170),
        '10': lambda: torpedo('depthHeadingAutopilot', 30, 50, 1525, 0.5, 170),
        # '11': lambda: OpenRig(mode="free", decay_axis="pitch", decay_angle_deg=30, decay_rate0=0.0)
        '11': lambda: OpenRig(mode="step", dof=1, step_amplitude=5.5, step_time=2.0)  # 0..5 (X,Y,Z,roll,pitch,yaw)
    }

    if no in vehicleOptions:
        vehicle = vehicleOptions[no]()
        printVehicleinfo(vehicle, sampleTime, N)
    else:
        print('Error: Not a valid simulator option')
        sys.exit()

    # Configure side current if applicable
    maybe_enable_current(vehicle)

    # Main simulation loop 
    [simTime, simData] = simulate(N, sampleTime, vehicle)
    
    # 3D plots and animation
    plotVehicleStates(simTime, simData, 1)                    
    plotControls(simTime, simData, vehicle, 2)
    plot3D(simData, numDataPoints, FPS, filename, 3) 

    # Save results to Excel (meta includes current settings if present)
    export_to_excel(simTime, simData, vehicle, out_dir=".", sampleTime=sampleTime, N=N)

    # webbrowser.get(browser).open_new_tab('file://' + os.path.abspath(filename))
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
