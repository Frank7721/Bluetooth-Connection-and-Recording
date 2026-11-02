#!/usr/bin/env python3
"""
Record BMI160 accelerometer + gyroscope data from two MetaWear RG/RPro boards
simultaneously and write everything to a single CSV.

Install:   pip install metawear
Run:       python record_two_metawear.py

Edit DEVICES / DURATION_S below. Command-line overrides are optional:
           python record_two_metawear.py --duration 60 --out other.csv

CSV columns:
  host_time_s     - time.perf_counter() when the packet reached this computer,
                    relative to recording start (same clock for both boards)
  device_epoch_ms - timestamp reported by the SDK for that sample
  device          - label from DEVICES
  sensor          - 'acc' (g) or 'gyro' (deg/s)
  x, y, z
"""
# ======================================================================
# EDIT THESE EACH SESSION
# ======================================================================
SESSION_NAME = "session1"   # saves  Session Data\<SESSION_NAME>.csv
DURATION_S = 30.0           # recording length in seconds
START_DELAY_S = 5           # countdown (s) after connecting, so you can set the boards
                            # down and let go before recording begins
# ======================================================================

import argparse
import csv
import os
import threading
import time
from time import sleep

from mbientlab.warble import BleScanner
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import (
    FnVoid_VoidP_DataP,
    GyroBoschOdr,
    GyroBoschRange,
)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DEVICES = {
    "left":  "F3:FA:18:F7:61:32",   # board 1
    "right": "D4:12:C5:89:FE:ED",   # board 2
}

OUT_FILE = f"{SESSION_NAME}.csv"     # output CSV filename (from SESSION_NAME)
SESSION_DIR = r"C:\Users\Frank\OneDrive - University of Waterloo\01 University of Waterloo\01 University of Waterloo [BME]\01 Terms\4A\BME 461 [Capstone]\Code\Bluetooth Connection and Recording\Session Data"

SCAN_TIMEOUT_S = 30.0               # how long to look for the boards first
ABORT_IF_NOT_SEEN = False           # False = still try to connect even if the scan
                                    #         didn't see a board (often works anyway)
RESCAN_S = 8.0                      # short re-scan between failed connect attempts
CONNECT_RETRIES = 5

ACC_ODR_HZ = 100.0                  # BMI160: 0.78 ... 1600 Hz
ACC_RANGE_G = 16.0                  # 2, 4, 8, or 16
GYRO_ODR = GyroBoschOdr._100Hz      # _25Hz ... _3200Hz
GYRO_RANGE = GyroBoschRange._500dps # _125dps ... _2000dps
# ----------------------------------------------------------------------------


def scan_for_boards(macs, timeout):
    """
    Windows can only open a BLE device by address after it has recently seen
    it advertising. Scanning first makes that happen. Returns the set of
    addresses that were NOT seen.
    """
    wanted = {m.upper() for m in macs}
    seen = set()

    def handler(result):
        mac = result.mac.upper()
        if mac in wanted:
            seen.add(mac)

    print(f"Scanning up to {timeout:.0f} s for {len(wanted)} board(s) ...")
    BleScanner.set_handler(handler)
    BleScanner.start()
    t_end = time.time() + timeout
    while time.time() < t_end and seen != wanted:
        sleep(0.25)
    BleScanner.set_handler(None)
    BleScanner.stop()

    for m in sorted(seen):
        print(f"  found {m}")
    return wanted - seen


class Recorder:
    """Shared CSV writer; callbacks from both boards are serialized with a lock."""

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["host_time_s", "device_epoch_ms", "device", "sensor", "x", "y", "z"]
        )
        self.lock = threading.Lock()
        self.t0 = None
        self.counts = {}

    def write(self, label, sensor, epoch_ms, x, y, z):
        if self.t0 is None:
            return
        t = time.perf_counter() - self.t0
        with self.lock:
            if self.file.closed:          # late packet after recording stopped
                return
            self.writer.writerow([f"{t:.6f}", epoch_ms, label, sensor, x, y, z])
            key = (label, sensor)
            self.counts[key] = self.counts.get(key, 0) + 1

    def close(self):
        with self.lock:
            self.file.flush()
            self.file.close()


class Board:
    def __init__(self, label, mac, recorder=None):
        self.label = label
        self.mac = mac
        self.rec = recorder
        self.device = MetaWear(mac)
        self.board = self.device.board
        self.connected = False
        self.configured = False
        self.started = False
        # Keep references to the C callbacks so they aren't garbage collected
        self._acc_cb = FnVoid_VoidP_DataP(self._on_acc)
        self._gyro_cb = FnVoid_VoidP_DataP(self._on_gyro)

    # --- callbacks -----------------------------------------------------------
    def _on_acc(self, ctx, data):
        if self.rec is None:
            return
        v = parse_value(data)
        self.rec.write(self.label, "acc", data.contents.epoch, v.x, v.y, v.z)

    def _on_gyro(self, ctx, data):
        if self.rec is None:
            return
        v = parse_value(data)
        self.rec.write(self.label, "gyro", data.contents.epoch, v.x, v.y, v.z)

    # --- lifecycle -----------------------------------------------------------
    def connect(self):
        last_err = None
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                print(f"[{self.label}] connecting to {self.mac} "
                      f"(attempt {attempt}/{CONNECT_RETRIES}) ...")
                self.device.connect()
                self.connected = True
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[{self.label}] connect failed: {e}")
                scan_for_boards([self.mac], RESCAN_S)   # let Windows see it again
                sleep(1.0)
        if not self.connected:
            raise RuntimeError(
                f"Could not connect to {self.label} ({self.mac}): {last_err}"
            )

        # Short connection interval = more BLE throughput (important for 2 boards)
        libmetawear.mbl_mw_settings_set_connection_parameters(
            self.board, 7.5, 7.5, 0, 6000
        )
        sleep(1.5)
        print(f"[{self.label}] connected")

    def configure(self):
        b = self.board
        # Accelerometer
        libmetawear.mbl_mw_acc_set_odr(b, ACC_ODR_HZ)
        libmetawear.mbl_mw_acc_set_range(b, ACC_RANGE_G)
        libmetawear.mbl_mw_acc_write_acceleration_config(b)
        acc_sig = libmetawear.mbl_mw_acc_get_acceleration_data_signal(b)
        libmetawear.mbl_mw_datasignal_subscribe(acc_sig, None, self._acc_cb)

        # Gyroscope (BMI160)
        libmetawear.mbl_mw_gyro_bmi160_set_odr(b, GYRO_ODR)
        libmetawear.mbl_mw_gyro_bmi160_set_range(b, GYRO_RANGE)
        libmetawear.mbl_mw_gyro_bmi160_write_config(b)
        gyro_sig = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(b)
        libmetawear.mbl_mw_datasignal_subscribe(gyro_sig, None, self._gyro_cb)
        self.configured = True

    def start(self):
        b = self.board
        libmetawear.mbl_mw_acc_enable_acceleration_sampling(b)
        libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(b)
        libmetawear.mbl_mw_acc_start(b)
        libmetawear.mbl_mw_gyro_bmi160_start(b)
        self.started = True

    def stop(self):
        b = self.board
        if self.started:
            libmetawear.mbl_mw_acc_stop(b)
            libmetawear.mbl_mw_gyro_bmi160_stop(b)
            libmetawear.mbl_mw_acc_disable_acceleration_sampling(b)
            libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(b)
            self.started = False
        if self.configured:
            acc_sig = libmetawear.mbl_mw_acc_get_acceleration_data_signal(b)
            gyro_sig = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(b)
            libmetawear.mbl_mw_datasignal_unsubscribe(acc_sig)
            libmetawear.mbl_mw_datasignal_unsubscribe(gyro_sig)
            self.configured = False

    def disconnect(self):
        if not self.connected:
            return
        try:
            self.device.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[{self.label}] disconnect warning: {e}")
        self.connected = False
        print(f"[{self.label}] disconnected")


def main():
    # The MetaWear SDK creates a ".metawear" cache folder in the current working
    # directory. Run from the script's own folder so it never lands somewhere
    # protected (e.g. C:\Windows\System32).
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Optional command-line overrides; with no arguments the values from the
    # config block at the top of the file are used.
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=DURATION_S, help="seconds to record")
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()

    # Save into the Session Data folder (created if missing) unless an absolute
    # path was given with --out
    if not os.path.isabs(args.out):
        os.makedirs(SESSION_DIR, exist_ok=True)
        args.out = os.path.join(SESSION_DIR, args.out)

    # Make sure Windows has seen both boards advertising before we connect
    missing = scan_for_boards(DEVICES.values(), SCAN_TIMEOUT_S)
    if missing:
        print("\nScan did not see these boards advertising:")
        for m in sorted(missing):
            print(f"  {m}")
        print("Check: board awake (press its button; LED should blink), nothing "
              "else connected to it, within ~1 m, Windows Bluetooth on, MAC correct.")
        if ABORT_IF_NOT_SEEN:
            return
        print("Trying to connect anyway ...\n")

    rec = Recorder(args.out)
    boards = [Board(label, mac, rec) for label, mac in DEVICES.items()]

    try:
        # Connect + configure each board (sequential is more reliable for BLE)
        for b in boards:
            b.connect()
        for b in boards:
            b.configure()

        if START_DELAY_S > 0:
            print("\nBoards are ready. Place them where they will start and let go - "
                  "keep them STILL for the first second or two of recording.")
            for k in range(int(START_DELAY_S), 0, -1):
                print(f"  recording starts in {k} ...")
                sleep(1)

        # Start recording clock, then start both boards back-to-back
        rec.t0 = time.perf_counter()
        for b in boards:
            b.start()
        print(f"Recording for {args.duration:.0f} s ... (Ctrl+C to stop early)")

        try:
            sleep(args.duration)
        except KeyboardInterrupt:
            print("\nStopping early.")

    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {e}")

    finally:
        for b in boards:
            try:
                b.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[{b.label}] stop warning: {e}")
        sleep(0.5)
        for b in boards:
            b.disconnect()
        rec.close()

    print(f"\nSaved {args.out}")
    for (label, sensor), n in sorted(rec.counts.items()):
        print(f"  {label:>8s} {sensor:<5s} {n} samples")


if __name__ == "__main__":
    main()
