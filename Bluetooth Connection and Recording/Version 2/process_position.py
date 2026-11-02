#!/usr/bin/env python3
"""
Estimate board POSITION on a flat plane (dead reckoning) from session.csv and
plot the X-Y path of each board.

Usage:   python process_position.py              (uses SESSION_NAME)
         python process_position.py trial2.csv

Reads  from SESSION_DIR  (imported from process_orientation.py)
Writes to   POSITION_DIR: <session>_position.csv + <session>_*_xy.png etc.

How it works
  1. Orientation (roll/pitch/yaw) from accel+gyro, as in process_orientation.py
  2. Rotate each accel sample into the START frame and subtract gravity
  3. Detect "stationary" moments (gyro ~0 and accel steady)
  4. Integrate accel -> velocity -> position, forcing velocity to 0 whenever the
     board is stationary (zero-velocity update, ZUPT) and removing the drift
     built up during each movement segment

Coordinate frame
  Origin (0,0) = where the board starts.  Start it FLAT and still for ~1 s.
  x = direction the board's x-axis pointed at the start
  y = 90 deg counter-clockwise from x, as seen from above
  So the plot is a top-down map.

Honest limits
  * Position from an IMU always drifts. Expect good shape and relative distances
    over short (seconds), pause-separated moves; expect error to grow with long
    continuous movement.
  * Yaw drifts (no magnetometer), which slowly rotates/curves the path.
  * Saturated sensors (gyro at its range limit) or dropped BLE samples corrupt
    the path. Record with GYRO_RANGE = _2000dps.
"""
# ======================================================================
# EDIT THESE EACH SESSION
# ======================================================================
SESSION_NAME = "session1"   # reads  Session Data\<SESSION_NAME>.csv
# ======================================================================

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_orientation import (   # same folder as this script
    BIAS_WINDOW_S,
    find_still_start,
    FS,
    SESSION_DIR,
    mahony,
    uniform_series,
)

# ------------------------------- settings -----------------------------------
INPUT_FILE = f"{SESSION_NAME}.csv"   # looked up in SESSION_DIR
POSITION_DIR = r"C:\Users\Frank\OneDrive - University of Waterloo\01 University of Waterloo\01 University of Waterloo [BME]\01 Terms\4A\BME 461 [Capstone]\Code\Bluetooth Connection and Recording\Processing Data\Positioning"

SHOW_PLOTS = True        # open plot windows (PNGs are always saved too)
FLAT_PLANE = True        # ignore vertical motion (you are sliding on a table)

STILL_WINDOW_S = 0.30    # window used to decide "stationary"
GYRO_STILL_DPS = 6.0     # mean |gyro| below this  -> candidate stationary
LIN_STILL_MPS2 = 0.25    # mean |gravity-removed accel| below this (m/s^2) -> stationary
PASS1_KP = 0.3           # weak gravity correction for the first (stillness-finding) pass
STILL_KP = 2.0           # gravity correction strength while stationary (pass 2)
G = 9.80665              # m/s^2 per g
# -----------------------------------------------------------------------------


def rotate(q, v):
    """Rotate vectors v (n,3) by quaternions q (n,4: w,x,y,z). Body -> start frame."""
    w, x, y, z = q.T
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2*(y*y + z*z); r[:, 0, 1] = 2*(x*y - w*z);   r[:, 0, 2] = 2*(x*z + w*y)
    r[:, 1, 0] = 2*(x*y + w*z);     r[:, 1, 1] = 1 - 2*(x*x + z*z); r[:, 1, 2] = 2*(y*z - w*x)
    r[:, 2, 0] = 2*(x*z - w*y);     r[:, 2, 1] = 2*(y*z + w*x);   r[:, 2, 2] = 1 - 2*(x*x + y*y)
    return np.einsum("nij,nj->ni", r, v)


def detect_still(lin_mps2, gyro_dps):
    """Stationary = little rotation AND little gravity-removed acceleration."""
    n = max(3, int(STILL_WINDOW_S * FS))
    gmag = pd.Series(np.linalg.norm(gyro_dps, axis=1))
    lmag = pd.Series(np.linalg.norm(lin_mps2, axis=1))
    g_ok = gmag.rolling(n, center=True, min_periods=1).mean() < GYRO_STILL_DPS
    l_ok = lmag.rolling(n, center=True, min_periods=1).mean() < LIN_STILL_MPS2
    return (g_ok & l_ok).to_numpy().copy()


def quality_warnings(dev, gyr, still, v):
    """Print plain-language warnings when the data can't give a trustworthy path."""
    warn = []
    # BMI160 full-scale ranges (deg/s). Clipping = many samples pinned at the top.
    sat = False
    for rng_dps in (125, 250, 500, 1000, 2000):
        near_top = np.abs(gyr) >= 0.985 * rng_dps
        if near_top.sum() >= 5 and np.abs(gyr).max() <= 1.02 * rng_dps:
            sat = True
    if sat:
        warn.append("gyro appears to be saturating (flat-topped at its range limit) - "
                    "re-record with GYRO_RANGE = _2000dps or move more slowly")
    if still.mean() < 0.15:
        warn.append(f"board was stationary only {still.mean()*100:.0f}% of the time - "
                    "without pauses the drift cannot be corrected; pause ~0.5 s "
                    "between movements")
    if np.linalg.norm(v[:, :2], axis=1).max() > 5:
        warn.append("estimated peak speed > 5 m/s, which is not physical for sliding "
                    "on a table - treat the path as unreliable")
    for w in warn:
        print(f"[{dev}] WARNING: {w}")


def integrate_with_zupt(a, still, dt):
    """a: (n,3) m/s^2. Returns velocity, position (n,3)."""
    n = len(a)
    v = np.zeros((n, 3))
    moving = (~still).astype(int)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], moving, [0]])))
    for s, e in zip(edges[::2], edges[1::2]):          # each [s, e) is a moving run
        seg = np.cumsum(a[s:e], axis=0) * dt
        if e < n:                                       # run ends at a stop: v must be 0
            ramp = (np.arange(1, e - s + 1) / (e - s))[:, None]
            seg = seg - ramp * seg[-1]
        v[s:e] = seg
    p = np.cumsum(v, axis=0) * dt
    return v, p


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    if not os.path.isabs(path):
        path = os.path.join(SESSION_DIR, path)
    if not os.path.exists(path):
        sys.exit(f"Could not find {path}\nCheck SESSION_DIR / INPUT_FILE.")
    process(path, show_plots=SHOW_PLOTS)


def process(path, show_plots=False):
    """Process one session CSV (full path); results go to POSITION_DIR."""
    stem = os.path.splitext(os.path.basename(path))[0]   # prefix for output files
    df = pd.read_csv(path)
    os.makedirs(POSITION_DIR, exist_ok=True)
    dt = 1.0 / FS

    results, tables = {}, []
    for dev, d in df.groupby("device"):
        ta, acc = uniform_series(d[d.sensor == "acc"])
        tg, gyro = uniform_series(d[d.sensor == "gyro"])
        t = np.arange(max(ta[0], tg[0]), min(ta[-1], tg[-1]), dt)
        acc_g = np.column_stack([np.interp(t, ta, acc[:, k]) for k in range(3)])
        gyr = np.column_stack([np.interp(t, tg, gyro[:, k]) for k in range(3)])

        s0 = find_still_start(acc_g, gyr, dev)         # skip handling before it settles
        t, acc_g, gyr = t[s0:], acc_g[s0:], gyr[s0:]

        n0 = int(BIAS_WINDOW_S * FS)
        gyr = gyr - gyr[:n0].mean(axis=0)                   # remove gyro bias
        g0 = acc_g[:n0].mean(axis=0)   # gravity as measured at rest at the start

        # Horizontal-plane basis from the initial "up" direction
        up = g0 / np.linalg.norm(g0)
        ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = ref - np.dot(ref, up) * up
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)                               # CCW from e1 seen from above
        basis = np.column_stack([e1, e2, up])

        def plane_accel(q):
            lin = (rotate(q, acc_g) - g0) * G               # linear accel, m/s^2
            a = lin @ basis                                 # (x, y, vertical)
            if FLAT_PLANE:
                a[:, 2] = 0.0
            return a

        # Pass 1: gentle orientation estimate -> find the stationary moments
        q1 = mahony(acc_g, np.radians(gyr), dt, kp=PASS1_KP)
        still = detect_still(plane_accel(q1), gyr)
        still[:n0] = True                                   # start is stationary by definition

        # Pass 2: trust gravity ONLY while stationary (otherwise a push looks like
        # a tilt and the filter cancels the very acceleration we want to integrate)
        q_rel = mahony(acc_g, np.radians(gyr), dt, kp=STILL_KP, trust=still)
        a_plane = plane_accel(q_rel)
        v, p = integrate_with_zupt(a_plane, still, dt)

        quality_warnings(dev, gyr, still, v)
        tt = t - t[0]
        path_len = float(np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)))
        print(f"[{dev}] stationary {still.mean()*100:.0f}% of the time | "
              f"path length {path_len:.2f} m | end point "
              f"({p[-1,0]:.2f}, {p[-1,1]:.2f}) m | peak speed "
              f"{np.linalg.norm(v[:, :2], axis=1).max():.2f} m/s")
        results[dev] = {"t": tt, "p": p, "still": still}
        tables.append(pd.DataFrame({
            "time_s": tt, "device": dev,
            "x_m": p[:, 0], "y_m": p[:, 1], "z_m": p[:, 2],
            "vx_mps": v[:, 0], "vy_mps": v[:, 1], "vz_mps": v[:, 2],
            "stationary": still.astype(int),
        }))

    out_csv = os.path.join(POSITION_DIR, f"{stem}_position.csv")
    pd.concat(tables, ignore_index=True).to_csv(out_csv, index=False, float_format="%.5f")

    if not show_plots:
        plt.switch_backend("Agg")

    # --- X-Y path per board (top-down map), colored by time -------------------
    for dev, r in results.items():
        p, t = r["p"], r["t"]
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot(p[:, 0], p[:, 1], color="0.7", linewidth=1, zorder=1)
        sc = ax.scatter(p[:, 0], p[:, 1], c=t, s=6, cmap="viridis", zorder=2)
        ax.scatter(*p[0, :2], s=120, c="tab:green", marker="o", label="start (0, 0)", zorder=3)
        ax.scatter(*p[-1, :2], s=120, c="tab:red", marker="s", label="end", zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"{dev} - estimated X-Y path (top-down)")
        ax.grid(alpha=.3); ax.legend(loc="best")
        fig.colorbar(sc, ax=ax, label="time (s)", shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(POSITION_DIR, f"{stem}_{dev}_xy.png"), dpi=120)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(t, p[:, 0], label="x"); ax.plot(t, p[:, 1], label="y")
        still = r["still"]                                  # shade stationary periods
        edges = np.flatnonzero(np.diff(np.concatenate([[0], still.astype(int), [0]])))
        for s, e in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[s], t[min(e, len(t)) - 1], color="0.9", zorder=0)
        ax.set_xlabel("time (s)"); ax.set_ylabel("position (m)")
        ax.set_title(f"{dev} - position vs time (grey = detected stationary)")
        ax.grid(alpha=.3); ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(POSITION_DIR, f"{stem}_{dev}_position_time.png"), dpi=120)

    # --- Overlay of all boards ------------------------------------------------
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for dev, r in results.items():
            line, = ax.plot(r["p"][:, 0], r["p"][:, 1], label=dev)
            ax.scatter(*r["p"][0, :2], c=line.get_color(), marker="o", s=80)
            ax.scatter(*r["p"][-1, :2], c=line.get_color(), marker="s", s=80)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title("All boards - X-Y path (circle = start, square = end)")
        ax.grid(alpha=.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(POSITION_DIR, f"{stem}_all_xy.png"), dpi=120)

    print(f"Wrote {out_csv} and plots to {POSITION_DIR}")
    if show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
