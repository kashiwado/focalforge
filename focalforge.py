#!/usr/bin/env python3
"""
OpenCV Camera Calibration Visualizer
=====================================
Interactive Kannala-Brandt (fisheye) intrinsic calibration and
XYZ + RPY extrinsic calibration visualizer using a webcam feed.

Projection model: Kannala-Brandt with 4 radial distortion coefficients.
Extrinsic model: 3D translation (X, Y, Z meters) + roll-pitch-yaw rotation.

Visualization:
  - Webcam feed as background
  - Semitransparent ground plane overlay (Z=0 in world space)
  - Opaque red horizontal lines at 5m intervals (Y=5..40m, X=-3..3m)
  - Opaque red vertical lines at X=-3m and X=+3m, Y=5..40m at 5m steps
  - All lines projected through the Kannala-Brandt model so they align
    with the distorted camera image.

Keyboard interface:
  Main menu:
    A / D      - switch between Intrinsic / Extrinsic menus
    Enter      - enter the highlighted menu
    Q / ESC    - quit (ESC also goes back from sub-menus)

  Intrinsic / Extrinsic sub-menus:
    W / S      - move between parameters (non-edit mode)
                 OR adjust the selected parameter value (edit mode)
    Enter      - toggle edit mode (activate parameter / confirm value)
    A / ESC    - back to main menu

  In edit mode the parameter name turns yellow and its value follows
  the live adjustment; pressing Enter locks the value back into the
  calibration structure.

  Note: cv2.waitKey does not reliably return arrow-key codes on this
  platform (arrow keys return 0), so WASD is used instead of the arrow
  keys.

Camera configuration can be loaded from a JSON file, passed as --config:

    python3 camera_calibration.py --config calib.json
    python3 camera_calibration.py 0 --config calib.json   # 0 = camera index

JSON format (all fields optional — missing keys keep their built-in
defaults):

    {
      "intrinsic": {
        "width": 1920,
        "height": 1080,
        "fx": 900.0,
        "fy": 900.0,
        "cx": 320.0,
        "cy": 240.0,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0
      },
      "extrinsic": {
        "x": 0.0,
        "y": 1.5,
        "z": 0.0,
        "roll": 0.0,
        "pitch": -15.0,
        "yaw": 0.0
      }
    }

  Intrinsic:
    fx, fy  — focal lengths in pixels
    cx, cy  — principal point in pixels
    k1..k4  — Kannala-Brandt radial distortion coefficients

  Extrinsic (camera pose in world space, Y-up convention):
    x, y, z   — camera position (y = height above ground)
    roll      — degrees, around Z (forward) axis
    pitch     — degrees, around X (right) axis; negative = nose down
    yaw       — degrees, around Y (up) axis; positive = turn left
"""

import cv2
import numpy as np
import math
import sys
import json
import argparse


# ---------------------------------------------------------------------------
# Kannala-Brandt projection
# ---------------------------------------------------------------------------

def kb_project(pt_3d, fx, fy, cx, cy, k1, k2, k3, k4):
    """Project a single 3-D point in **camera coordinates** to pixel (u, v).

    Camera frame:  X = right,  Y = DOWN (image),  Z = forward (optical axis).
    Points behind the camera plane (Z <= 1e-6) return ``None``.
    """
    X, Y, Z = pt_3d
    if Z <= 1e-6:
        return None

    x = X / Z
    y = Y / Z
    r = math.sqrt(x * x + y * y)

    if r < 1e-12:
        xd = yd = 0.0
    else:
        theta = math.atan(r)
        theta_d = theta * (1.0 + k1 * theta**2 + k2 * theta**4 +
                           k3 * theta**6 + k4 * theta**8)
        s = theta_d / r
        xd = s * x
        yd = s * y

    return (fx * xd + cx, fy * yd + cy)


# ---------------------------------------------------------------------------
# Roll-Pitch-Yaw -> rotation matrix  (world -> camera)
# ---------------------------------------------------------------------------
#
# World frame:  X = right,  Y = UP,  Z = forward (into scene).
# Camera frame: X = right,  Y = DOWN,  Z = forward (optical axis).
# Ground plane: Y = 0  (flat horizontal at world Y = 0).
# Camera position:  C = (ext_x, ext_y, ext_z)  — ext_y = height above ground.
#
# R = [X_c; Y_c; Z_c]  (rows are the camera axes expressed in world frame).
# P_c = R @ (P_w - C)
#
# We build R as a product of standard right-handed rotation matrices:
#
#   R = R_flip @ R_y(yaw) @ R_x(pitch) @ R_z(roll)
#
# Intrinsic rotation matrices:
#   R_y(yaw)   — rotate around Y (up)      by +yaw   (camera turns left)
#   R_x(pitch) — rotate around X (right)  by +pitch  (nose up when > 0)
#   R_z(roll)  — rotate around Z (forward) by +roll   (clockwise bank when > 0)
#
# R_flip = diag(1, -1, 1) maps world Y-up → camera Y-down (image convention).
# Matrix order is right-to-left: yaw applied first, then pitch, then roll,
# then Y-flip — matching the intrinsic yaw → pitch → roll rotation sequence.
# ---------------------------------------------------------------------------


def rpy_to_R(roll_deg, pitch_deg, yaw_deg):
    """Build the world→camera rotation matrix R.

    World frame:   X = right,  Y = UP,   Z = forward (into scene).
    Camera frame:  X = right,  Y = DOWN,  Z = forward (optical axis).
    Ground plane:  Y = 0.  Camera at C = (ext_x, ext_y, ext_z).

    R = [X_c; Y_c; Z_c]  (rows = camera axes in world frame).
    P_c = R @ (P_w - C)
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)

    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    Ry = np.array([[cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]])

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cp, -sp],
                   [0.0, sp, cp]])

    Rz = np.array([[cr, -sr, 0.0],
                   [sr, cr, 0.0],
                   [0.0, 0.0, 1.0]])

    R_flip = np.array([[1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, 1.0]])

    # Intrinsic yaw → pitch → roll, then Y-flip for camera down direction.
    return R_flip @ Ry @ Rx @ Rz


def world_to_camera(pt_w, R, T):
    """Transform a world point to camera coordinates:  P_c = R @ (P_w - T)."""
    return R @ (np.array(pt_w) - np.array(T))


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class CalibrationVisualizer:
    def __init__(self, camera_index=0, config_path=None):
        # ---- intrinsic (Kannala-Brandt) ------------------------------------
        self.width = 640
        self.height = 480
        self.fx = 600
        self.fy = 600
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.k4 = 0.0

        # ---- extrinsic (camera pose in world) ------------------------------
        # World convention: X = right, Y = UP, Z = forward (into scene).
        # Ground plane is Y = 0.  Camera sits at (ext_x, ext_y, ext_z).
        # Pitch tilts the camera around its X axis: negative = nose down.
        self.ext_x   = 0.0
        self.ext_y   = 1.5          # camera height above ground
        self.ext_z   = 0.0
        self.roll    = 0.0          # degrees  (around Z / forward axis)
        self.pitch   = -15.0        # degrees  (around X / right axis; <0 = down)
        self.yaw     = 0.0          # degrees  (around Y / up axis; >0 = left)

        # ---- load user-provided overrides from JSON (optional) --------------
        if config_path:
            self.load_config(config_path)

        # Try to initialize the video device or stream
        if sys.platform == 'win32':
            self.cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)  # Windows requires DirectShow support
        else:
            self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"ERROR: Cannot open camera {camera_index}")

        original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if config_path:
            if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width) or not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height):
                raise RuntimeError(f"ERROR: Failed to configure video stream to use resolution {self.width}x{self.height} (was {original_width}x{original_height}). Verify the calibration file if used.")
        else:
            self.width = original_width
            self.height = original_height

        if self.width < 2 or self.height < 2:
            self.width, self.height = 640, 480


        print(f"Initialized camera resolution: {self.width}x{self.height}")

        self.line_size = int(math.ceil(0.0032 * self.width))

        # ---- menu state ----------------------------------------------------
        self.menu_state   = "main"   # "main" | "intrinsic" | "extrinsic"
        self.sel_param    = 0
        self.editing      = False
        self.edit_value   = 0.0

        # Parameter catalogue: (label,  caller-settable_ref,  lo, hi, step, fmt)
        # The caller-settable_ref is a *lambda* so menus can ping the real
        # current value even after it was modified by another route.
        #     Field name             variable     min  max     step   precision
        self.intrinsic_params = [
            ("width",        lambda: self.width,  320, 6580,    2.0,  ".1f"),
            ("height",       lambda: self.height, 240, 4320,    2.0,  ".1f"),
            ("f_x",          lambda: self.fx,  10.0,  3000.0,  1.0,  ".1f"),
            ("f_y",          lambda: self.fy,  10.0,  3000.0,  1.0,  ".1f"),
            ("c_x",          lambda: self.cx,  1.0,   float(self.width),   1.0, ".1f"),
            ("c_y",          lambda: self.cy,  1.0,   float(self.height),  1.0, ".1f"),
            ("k1 (radial)",  lambda: self.k1, -2.0,   2.0,     0.001, ".5f"),
            ("k2 (radial)",  lambda: self.k2, -2.0,   2.0,     0.001, ".5f"),
            ("k3 (radial)",  lambda: self.k3, -2.0,   2.0,     0.001, ".5f"),
            ("k4 (radial)",  lambda: self.k4, -2.0,   2.0,     0.001, ".5f"),
        ]
        self.extrinsic_params = [
            ("X (m)",        lambda: self.ext_x,  -250.0,  25.0,   0.01,  ".2f"),
            ("Y (m)",        lambda: self.ext_y,  -250,    25.0,   0.01,  ".2f"),
            ("Z (m)",        lambda: self.ext_z,  0.1,     10.0,   0.01,  ".2f"),
            ("Roll (°)",     lambda: self.roll,  -180.0,  180.0,  0.1,   ".1f"),
            ("Pitch (°)",    lambda: self.pitch, -90.0,    90.0,   0.1,   ".1f"),
            ("Yaw (°)",      lambda: self.yaw,   -180.0,  180.0,  0.1,   ".1f"),
        ]

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    _PARAM_MAP = {
        # sensor and lens intrinsic parameters
        "width":  (lambda s, v: setattr(s, "width",  float(v)), "intrinsic"),
        "height": (lambda s, v: setattr(s, "height", float(v)), "intrinsic"),
        "fx":     (lambda s, v: setattr(s, "fx",     float(v)), "intrinsic"),
        "fy":     (lambda s, v: setattr(s, "fy",     float(v)), "intrinsic"),
        "cx":     (lambda s, v: setattr(s, "cx",     float(v)), "intrinsic"),
        "cy":     (lambda s, v: setattr(s, "cy",     float(v)), "intrinsic"),
        "k1":     (lambda s, v: setattr(s, "k1",     float(v)), "intrinsic"),
        "k2":     (lambda s, v: setattr(s, "k2",     float(v)), "intrinsic"),
        "k3":     (lambda s, v: setattr(s, "k3",     float(v)), "intrinsic"),
        "k4":     (lambda s, v: setattr(s, "k4",     float(v)), "intrinsic"),
        # camera installation extrinsic parameters
        "x":      (lambda s, v: setattr(s, "ext_x",  float(v)), "extrinsic"),
        "y":      (lambda s, v: setattr(s, "ext_y",  float(v)), "extrinsic"),
        "z":      (lambda s, v: setattr(s, "ext_z",  float(v)), "extrinsic"),
        "roll":   (lambda s, v: setattr(s, "roll",   float(v)), "extrinsic"),
        "pitch":  (lambda s, v: setattr(s, "pitch",  float(v)), "extrinsic"),
        "yaw":    (lambda s, v: setattr(s, "yaw",    float(v)), "extrinsic"),
    }

    def load_config(self, path):
        """Load calibration parameters from a JSON file.

        Only keys present in the file override the built-in defaults;
        missing keys keep their current value.

        Expected format::

            {
              "intrinsic": {"fx": 900.0, "fy": 900.0, "cx": 320, "cy": 240,
                            "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
              "intrinsic": {"width": 640, "height": 480,
                            "fx": 900.0, "fy": 900.0, "cx": 320, "cy": 240,
                            "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
              "extrinsic": {"x": 0.0, "y": 1.5, "z": 0.0,
                            "roll": 0.0, "pitch": -15.0, "yaw": 0.0}
            }
        """
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            print(f"WARNING: config file '{path}' not found — using defaults.",
                  file=sys.stderr)
            return
        except json.JSONDecodeError as exc:
            print(f"WARNING: could not parse JSON in '{path}': {exc}",
                  file=sys.stderr)
            print("         Using built-in defaults.", file=sys.stderr)
            return

        for section in ("intrinsic", "extrinsic"):
            if section not in data:
                continue
            for key, val in data[section].items():
                if key in self._PARAM_MAP:
                    setter, _ = self._PARAM_MAP[key]
                    setter(self, val)
                    print(f"  config: {section}.{key} = {val}")
                else:
                    print(f"  WARNING: unknown parameter '{section}.{key}' — ignored",
                          file=sys.stderr)
        print(f"Loaded configuration from {path}")


    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def project_world(self, X, Y, Z):
        """Project a world-point through the full extrinsic + KB pipeline.

        Returns None when the transformed point lies behind the camera
        plane (Z_cam <= 1e-6).  Behind-camera points simply aren't drawn;
        the line/grid builder already handles those gracefully.
        """
        R = rpy_to_R(self.roll, self.pitch, self.yaw)
        T = np.array([self.ext_x, self.ext_y, self.ext_z])
        P_c = world_to_camera((X, Y, Z), R, T)
        return kb_project(P_c, self.fx, self.fy, self.cx, self.cy,
                          self.k1, self.k2, self.k3, self.k4)

    # ------------------------------------------------------------------
    # Ground-plane drawing
    # ------------------------------------------------------------------

    def draw_scene(self, frame):
        """Draw ground plane + red measurement lines on *frame* (in place)."""
        overlay = frame.copy()

        # --- ground plane grid (semitransparent) ---------------------------
        # World:  X = lateral,  Y = UP,  Z = forward.  Ground plane = Y = 0.
        GRID_SPACING = 1.00          # metres per cell
        X_MIN, X_MAX = -8.0, 8.0    # lateral range
        Z_MIN, Z_MAX = 0.0, 45.0    # forward range

        xs = np.arange(X_MIN, X_MAX + 0.01, GRID_SPACING)
        zs = np.arange(Z_MIN, Z_MAX + 0.01, GRID_SPACING)

        # Pre-project every grid vertex (all on the ground plane, Y = 0)
        proj = {}
        for z in zs:
            for x in xs:
                pt = self.project_world(x, 0.0, z)
                proj[(x, z)] = pt  # may be None

        # Build quad polygons (only those where all 4 corners are visible)
        quads = []
        for i in range(len(zs) - 1):
            z0, z1 = zs[i], zs[i + 1]
            for j in range(len(xs) - 1):
                x0, x1 = xs[j], xs[j + 1]
                p00 = proj.get((x0, z0))
                p01 = proj.get((x1, z0))
                p10 = proj.get((x0, z1))
                p11 = proj.get((x1, z1))
                if None in (p00, p01, p10, p11):
                    continue
                quad = np.array([
                    [int(p00[0]), int(p00[1])],
                    [int(p01[0]), int(p01[1])],
                    [int(p11[0]), int(p11[1])],
                    [int(p10[0]), int(p10[1])],
                ], dtype=np.int32)
                quads.append(quad)

        if quads:
            cv2.polylines(overlay, quads, isClosed = True, color = (120, 0, 0), thickness = 4)

        # --- red measurement lines -----------------------------------------
        # Horizontal lines: at Z = 5,10,...,40 m, X from -3 to +3 m, on ground (Y=0)
        distances = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        for d in distances:
            p_left  = self.project_world(-3.0, 0.0, float(d))
            p_right = self.project_world( 3.0, 0.0, float(d))
            if p_left and p_right:
                cv2.line(overlay,
                         (int(p_left[0]),  int(p_left[1])),
                         (int(p_right[0]), int(p_right[1])),
                         (0, 0, 255), self.line_size)

        # Vertical lines: at X = -3 m and X = +3 m, from Z = 0 to 40 m, on ground (Y=0)
        for lat in (-3.0, 3.0):
            pts = []
            for d in distances:
                p = self.project_world(lat, 0.0, float(d))
                if p:
                    pts.append((int(p[0]), int(p[1])))
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(overlay, pts[i], pts[i + 1], (0, 0, 255), self.line_size)

        # Blend overlay back onto the frame
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame

    # ------------------------------------------------------------------
    # Menu / HUD drawing
    # ------------------------------------------------------------------

    def draw_menu(self, frame):
        h = self.height
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = 0.5
        thick = 1
        x = 10

        if self.menu_state == "main":
            y = 20
            cv2.putText(frame, "FocalForge Calibration Visualizer",
                        (x, y), font, 0.6, (255, 255, 255), self.line_size)
            y += 28
            cv2.putText(frame,
                        "A / D : switch menu   |   Enter : enter   |   Q / ESC : quit",
                        (x, y), font, 0.45, (200, 200, 200), 1)
            y += 28

            choices = [
                ("[I] Intrinsic  (Kannala-Brandt)", self.menu_state == "intrinsic"),
                ("[E] Extrinsic (XYZ + RPY)",        self.menu_state == "extrinsic"),
            ]
            for label, active in choices:
                color = (0, 255, 0) if active else (180, 180, 180)
                prefix = "> " if active else "  "
                cv2.putText(frame, prefix + label, (x, y),
                            font, fs, color, thick)
                y += 24
            y += 8
            cv2.putText(frame, "[Q] Quit", (x, y),
                        font, fs, (180, 180, 180), thick)

        else:   # intrinsic or extrinsic sub-menu
            params = (self.intrinsic_params
                      if self.menu_state == "intrinsic"
                      else self.extrinsic_params)
            title = ("Intrinsic Parameters"
                     if self.menu_state == "intrinsic" else
                     "Extrinsic Parameters")
            y = 20
            cv2.putText(frame, title, (x, y),
                        font, 0.55, (255, 255, 255), 2)
            y += 24
            cv2.putText(frame,
                        "W / S : highlight / adjust   |   Enter : edit   |   A / ESC : back",
                        (x, y), font, 0.4, (200, 200, 200), 1)
            y += 26

            for i, (label, getter, lo, hi, step, fmt) in enumerate(params):
                active = (i == self.sel_param)
                color = (0, 255, 0) if active else (180, 180, 180)
                if self.editing and active:
                    color = (0, 255, 255)   # yellow when editing

                prefix = "> " if active else "  "

                val = self.edit_value if (self.editing and active) else getter()
                val_str = f"{val:{fmt}}"
                line = f"{prefix}{label}: {val_str}"
                cv2.putText(frame, line, (x, y),
                            font, fs, color, thick)

                if active:
                    rng = f"    [{lo:.2f} .. {hi:.2f}]"
                    cv2.putText(frame, rng, (x, y + 14),
                                font, fs * 0.7, (150, 150, 150), 1)
                    if self.editing and active:
                        cv2.putText(frame,
                                    "EDITING — W/S to adjust, Enter to confirm",
                                    (x, y + 32), font, 0.4, (0, 255, 255), 1)
                        y += 50
                        continue
                y += 22

        return frame

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def handle_key(self, key_code):
        """Return False to quit the application loop.

        Single-character keys (W/A/S/D, I/E, Q, Enter) are used throughout
        because cv2.waitKey does not reliably return arrow-key codes on this
        platform (arrow keys return 0).
        """
        # -- ESC / Q -------------------------------------------------------
        if key_code in (27, ord('q'), ord('Q')):
            if self.menu_state == "main":
                return False
            self.menu_state = "main"
            self.sel_param  = 0
            self.editing    = False
            return True

        # -- Main-menu shortcuts (only when in the top-level menu) ----------
        if self.menu_state == "main" and key_code <= 255:
            if key_code in (ord('i'), ord('I')):
                self.menu_state = "intrinsic"
                self.sel_param  = 0
                self.editing    = False
                return True
            if key_code in (ord('e'), ord('E')):
                self.menu_state = "extrinsic"
                self.sel_param  = 0
                self.editing    = False
                return True

        # -- WASD navigation (substitutes for arrow keys) -------------------
        # Up
        if key_code in (ord('w'), ord('W')):
            self._on_up()
            return True
        # Down
        if key_code in (ord('s'), ord('S')):
            self._on_down()
            return True
        # Left
        if key_code in (ord('a'), ord('A')):
            self._on_left()
            return True
        # Right
        if key_code in (ord('d'), ord('D')):
            self._on_right()
            return True

        # -- Enter / Return -------------------------------------------------
        if key_code in (13, 10):
            if self.menu_state in ("intrinsic", "extrinsic"):
                if self.editing:
                    self._apply_edit()
                    self.editing = False
                else:
                    self.editing  = True
                    params = (self.intrinsic_params
                              if self.menu_state == "intrinsic"
                              else self.extrinsic_params)
                    self.edit_value = params[self.sel_param][1]()
            return True

        # -- Anything else is ignored --------------------------------------
        return True

    # ---- menu navigation actions ----------------------------------------

    def _params(self):
        return (self.intrinsic_params
                if self.menu_state == "intrinsic"
                else self.extrinsic_params)

    def _on_up(self):
        if self.menu_state == "main":
            return
        if self.editing:
            self._adjust_selected(+1)
        else:
            n = len(self._params())
            self.sel_param = (self.sel_param - 1) % n

    def _on_down(self):
        if self.menu_state == "main":
            return
        if self.editing:
            self._adjust_selected(-1)
        else:
            n = len(self._params())
            self.sel_param = (self.sel_param + 1) % n

    def _on_right(self):
        if self.menu_state == "main":
            # toggle sub-menu highlight
            if self.sel_param == 0:
                self.sel_param = 1
            else:
                self.sel_param = 0
            print(f"[ _on_right] main: toggled sel_param -> {self.sel_param}")
        else:
            # enter the other sub-menu?  No — Left goes back; Right does nothing
            pass

    def _on_left(self):
        if self.menu_state == "main":
            if self.sel_param == 0:
                self.sel_param = 1
            else:
                self.sel_param = 0
            print(f"[ _on_left] main: toggled sel_param -> {self.sel_param}")
        else:
            self.menu_state = "main"
            self.sel_param  = 0
            self.editing    = False
            print("[ _on_left] sub-menu: back to main")

    def _adjust_selected(self, direction):
        params = self._params()
        label, getter, lo, hi, step, fmt = params[self.sel_param]
        self.edit_value = self.edit_value + direction * step
        self.edit_value = max(lo, min(hi, self.edit_value))

    def _apply_edit(self):
        params = self._params()
        label, ref, lo, hi, step, fmt = params[self.sel_param]
        val = self.edit_value

        if self.menu_state == "intrinsic":
            if label.startswith("width"): self.width = val
            if label.startswith("height"):self.height = val
            if label.startswith("f_x"):   self.fx = val
            elif label.startswith("f_y"): self.fy = val
            elif label.startswith("c_x"): self.cx = val
            elif label.startswith("c_y"): self.cy = val
            elif label.startswith("k1"):  self.k1 = val
            elif label.startswith("k2"):  self.k2 = val
            elif label.startswith("k3"):  self.k3 = val
            elif label.startswith("k4"):  self.k4 = val
        else:   # extrinsic
            if label.startswith("X ("):     self.ext_x = val
            elif label.startswith("Y ("):   self.ext_y = val
            elif label.startswith("Z ("):   self.ext_z = val
            elif label == "Roll (°)":       self.roll  = val
            elif label == "Pitch (°)":      self.pitch = val
            elif label == "Yaw (°)":        self.yaw   = val

    # ------------------------------------------------------------------
    # Live info strip
    # ------------------------------------------------------------------

    def draw_info(self, frame):
        txt = (
            f"width={self.width} height={self.height} "
            f"fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f} | "
            f"k1={self.k1:.4f} k2={self.k2:.4f} "
            f"k3={self.k3:.4f} k4={self.k4:.4f} | "
            f"X={self.ext_x:.2f} Y={self.ext_y:.2f} Z={self.ext_z:.2f} | "
            f"R={self.roll:.1f} P={self.pitch:.1f} Yw={self.yaw:.1f}"
        )
        cv2.putText(frame, txt, (10, int(self.height) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print("=" * 60)
        print("  Camera Calibration Visualizer")
        print("=" * 60)
        print("  Webcam feed  +  Kannala-Brandt projection  +  ground plane")
        print()
        print("  Main menu:")
        print("    A / D  (Left/Right)  — highlight Intrinsic / Extrinsic")
        print("    Enter                 — enter highlighted menu")
        print("    Q / ESC               — quit (ESC also goes back from sub-menus)")
        print()
        print("  Sub-menus (Intrinsic / Extrinsic):")
        print("    W / S  (Up/Down)      — highlight parameter OR adjust value")
        print("    Enter                 — toggle edit mode (activate / confirm)")
        print("    A / ESC               — back to main menu")
        print()
        print("  Press a key to start ...")
        print("=" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("WARNING: camera frame read failed — retrying ...", file=sys.stderr)
                cv2.waitKey(50)
                continue

            self.draw_scene(frame)
            self.draw_menu(frame)
            self.draw_info(frame)

            cv2.imshow("Camera Calibration Visualizer", frame)

            key = cv2.waitKey(1)
            # print(f"[main] waitKey -> key={key} (repr={key!r})")  # debug
            if key == -1 or key == 0:
                continue

            if not self.handle_key(key):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Bye.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":

    default_video_stream = ""
    if sys.platform == 'win32':
        default_video_stream = 0
    else:
        default_video_stream = "/dev/video0"

    parser = argparse.ArgumentParser(
        description="OpenCV Camera Calibration Visualizer — Kannala-Brandt intrinsic + 3D ground-plane extrinsic",
        usage="%(prog)s [CAMERA_INDEX (windows) or CAMERA_PATH (POSIX)] [--config JSON_PATH]")
    parser.add_argument("camera", nargs="?", type=str, default=default_video_stream,
                        help=f"video device index (default: {default_video_stream})")
    parser.add_argument("--config", "-c", type=str, default=None, metavar="PATH",
                        help="JSON file with intrinsic/extrinsic parameters")
    args = parser.parse_args()


    if args.config:
        print(f"Loading config from {args.config} ...")
    try:
        app = CalibrationVisualizer(camera_index=args.camera, config_path=args.config)
        app.run()
    except Exception as e:
        print(e)
        parser.print_usage()
