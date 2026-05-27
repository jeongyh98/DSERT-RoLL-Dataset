import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pickle

# -----------------------------
# Helpers
# -----------------------------

def is_file(path: str) -> bool:
    return os.path.isfile(path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_rvec_R(r_like: np.ndarray):
    r_like = np.asarray(r_like)
    if r_like.shape == (3, 3):
        rvec, _ = cv2.Rodrigues(r_like)
        R = r_like
    else:
        rvec = r_like.reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
    return rvec, R

# -----------------------------
# Event visualizer
# -----------------------------

def visualize_event(event_array):
    """Render event data as an RGB image (red=positive, blue=negative polarity).

    Note: output resolution is hardcoded to (704, 1152) matching this dataset's
    event camera specification.
    """
    vis_tensor_ = np.ones((704, 1152, 3), np.uint8) * 255
    xs = event_array[:, 0]
    ys = event_array[:, 1]
    ps = event_array[:, 2]
    vis_tensor_[ys[ps == 1], xs[ps == 1], 0] = 255
    vis_tensor_[ys[ps == 1], xs[ps == 1], 1] = 0
    vis_tensor_[ys[ps == 1], xs[ps == 1], 2] = 0
    vis_tensor_[ys[ps == 0], xs[ps == 0], 0] = 0
    vis_tensor_[ys[ps == 0], xs[ps == 0], 1] = 0
    vis_tensor_[ys[ps == 0], xs[ps == 0], 2] = 255
    return vis_tensor_

# -----------------------------
# Projection core
# -----------------------------

def project_points_to_image(points_xyz: np.ndarray, color_vals: np.ndarray,
                            R_like: np.ndarray, tvec: np.ndarray,
                            K: np.ndarray, dist: np.ndarray,
                            w: int, h: int,
                            z_min: float = 1.0):
    rvec, R = to_rvec_R(R_like)
    cam_pts = (R @ points_xyz.T + tvec.reshape(3, 1)).T

    img_pts, _ = cv2.projectPoints(points_xyz, rvec, tvec.reshape(3, 1), K, dist)
    img_pts = img_pts[:, 0]

    valid = (
        (img_pts[:, 0] >= 1) & (img_pts[:, 0] < w - 1) &
        (img_pts[:, 1] >= 1) & (img_pts[:, 1] < h - 1) &
        (cam_pts[:, 2] > z_min)
    )

    if np.any(valid):
        c = color_vals[valid].astype(np.float32)
        c = np.clip(c, np.percentile(c, 2), np.percentile(c, 98))
        c = (c - c.min()) / max(1e-6, (c.max() - c.min()))
        return img_pts[valid], c, valid
    else:
        return np.empty((0, 2)), np.empty((0,)), valid

# -----------------------------
# 3D BBox utils
# -----------------------------

def clip_line_to_rect(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    """Liang-Barsky clipping. Returns clipped segment or None if fully outside."""
    dx, dy = x2 - x1, y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        elif pi < 0:
            t0 = max(t0, qi / pi)
        else:
            t1 = min(t1, qi / pi)
    if t0 > t1:
        return None
    return x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy


def get_3d_box_projected_corners(vehicle_to_image, box_to_vehicle):
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)
    vertices = np.empty([2, 2, 2, 2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                v = np.array([(k - 0.5), (l - 0.5), (m - 0.5), 1.0])
                v = np.matmul(box_to_image, v)
                vertices[k, l, m, :] = [v[0] / v[2], v[1] / v[2]]
    return vertices

# -----------------------------
# Viewer
# -----------------------------

class MultiSensorProjectionViewer:
    """Visualizes all sensor combinations in a 3×6 grid (rows: RGB/Event/Thermal, cols: Livox/Ouster/Radar × L/R).

    Loads data from a session's label.pkl and projects LiDAR/Radar point clouds
    onto each camera image. Optionally overlays 3D bounding boxes (Livox frame).
    """

    SENSOR_CONFIGS = {
        "Livox": {"pc_glob": "LIDAR_LIVOX_Tilted/*.npy", "color_axis": "x"},
        "Ouster": {"pc_glob": "LIDAR_OUSTER_Tilted_90_degree/*.npy", "color_axis": "x"},
        "Radar": {"pc_glob": "RADAR_Tilted/*.npy", "color_axis": "y"},
    }

    # 3×6 grid: row = camera type (RGB/Event/Thermal), col = sensor × side (L/R)
    GRID_ORDER = [
        ("RGB_L", "Livox"), ("RGB_R", "Livox"), ("RGB_L", "Ouster"), ("RGB_R", "Ouster"), ("RGB_L", "Radar"), ("RGB_R", "Radar"),
        ("Event_L", "Livox"), ("Event_R", "Livox"), ("Event_L", "Ouster"), ("Event_R", "Ouster"), ("Event_L", "Radar"), ("Event_R", "Radar"),
        ("Thermal_L", "Livox"), ("Thermal_R", "Livox"), ("Thermal_L", "Ouster"), ("Thermal_R", "Ouster"), ("Thermal_L", "Radar"), ("Thermal_R", "Radar"),
    ]

    def __init__(self, folder_path: str, root_path: str):
        self.folder_path = folder_path
        self.root_path = root_path
        pkl_path = os.path.join(folder_path, 'label.pkl')
        self.pkl_path = pkl_path
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as fr:
                data = pickle.load(fr)
            self.pkl_exist = True
            self.data = data['info']
            self.meta = data['meta']
            self.num_frames = data['meta']['sequence_len']
            self.data_sensor_keymap = {
                "RGB_L": "rgb_left_path", "RGB_R": "rgb_right_path",
                "Event_L": "event_left_path", "Event_R": "event_right_path",
                "Thermal_L": "thermal_left_path", "Thermal_R": "thermal_right_path",
            }

            self.K = {cam: np.asarray(self.meta['calibration'][cam]["intrinsic"], dtype=np.float64)
                      for cam in ["RGB_L", "RGB_R", "Event_L", "Event_R", "Thermal_L", "Thermal_R"]}
            self.dist = {cam: np.zeros((4, 1), dtype=np.float64) for cam in self.K}

            self.extr = {}
            for sensor in ["Livox", "Ouster", "Radar"]:
                ext_dict = {}
                sd = self.meta['calibration'][sensor]
                for cam in ["RGB_L", "RGB_R", "Event_L", "Event_R", "Thermal_L", "Thermal_R"]:
                    T = np.asarray(sd[cam], dtype=np.float64)
                    ext_dict[cam] = (T[:3, :3], T[:3, 3])
                self.extr[sensor] = ext_dict

            self.idx = 0
            self._build_figure()
        else:
            self.pkl_exist = False

    def _build_figure(self):
        """Create the matplotlib figure with aspect-ratio-aware panel sizing.

        Reads the first frame to determine each camera row's actual aspect ratio,
        then computes figure dimensions so every panel is the same column width.
        """
        sample = self._load_images_for_index(0)
        row_asp = []
        for cam in ["RGB_L", "Event_L", "Thermal_L"]:
            img = sample.get(cam)
            row_asp.append(img.shape[0] / img.shape[1] if img is not None else 3 / 4)

        col_w   = 3.0          # column width in inches
        col_gap = 0.08         # horizontal gap between columns
        row_gap = 0.3 / 2.54   # vertical gap between rows (0.3 cm)
        title_h = (11 + 4) / 72  # axes title height (font 11pt + pad 4pt)
        suptitle_h = 0.8
        bot_margin = 0.05
        row_h = [col_w * a for a in row_asp]

        fig_w = 6 * col_w + 5 * col_gap
        fig_h = (bot_margin
                 + row_h[2] + row_gap + title_h
                 + row_h[1] + row_gap + title_h
                 + row_h[0] + title_h
                 + suptitle_h)

        row_bot = [0.0, 0.0, 0.0]
        row_bot[2] = bot_margin
        row_bot[1] = row_bot[2] + row_h[2] + row_gap + title_h
        row_bot[0] = row_bot[1] + row_h[1] + row_gap + title_h

        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.axs = []
        for r in range(3):
            for c in range(6):
                ax = self.fig.add_axes([
                    (c * (col_w + col_gap)) / fig_w,
                    row_bot[r] / fig_h,
                    col_w / fig_w,
                    row_h[r] / fig_h,
                ])
                self.axs.append(ax)

    def _load_images_for_index(self, idx: int):
        """Load all camera images for a given frame index. Returns dict keyed by camera name."""
        imgs = {}
        sens = self.data[idx].get("sensor", {})
        for cam, key in self.data_sensor_keymap.items():
            rel = sens.get(key)
            p = os.path.join(self.root_path, rel) if rel is not None else None
            img = None
            if p is not None and is_file(p):
                if p.lower().endswith((".jpg", ".jpeg", ".png")):
                    im = cv2.imread(p)
                    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                elif p.lower().endswith(".npz"):
                    try:
                        ev = np.load(p)
                        if 'event' in ev:
                            img = visualize_event(ev['event'])
                        elif 'image' in ev:
                            img = ev['image']
                    except Exception:
                        img = None
            imgs[cam] = img
        return imgs

    def _load_pointcloud_for_sensor(self, sensor: str, idx: int):
        """Load point cloud for the given sensor and frame. Returns (xyz, color) or None."""
        key_map = {"Livox": "livox_path", "Ouster": "ouster_path", "Radar": "radar_path"}
        sens = self.data[idx].get("sensor", {})
        rel = sens.get(key_map[sensor])
        if rel is None:
            return None
        p = os.path.join(self.root_path, rel)
        if not is_file(p):
            return None
        npy = np.load(p)
        if len(npy) > 0:
            xyz = np.vstack([npy['x'], npy['y'], npy['z']]).T.astype(np.float64)
            axis = self.SENSOR_CONFIGS[sensor]["color_axis"]
            color = npy['x'] if axis == 'x' else npy['y']
        else:
            return None
        return xyz, color

    def _draw_3d_bbox(self, ax, annos, K, extr, w, h):
        for cls, dim, cent, yaw in zip(annos['name'], annos['dimensions'], annos['location'], annos['heading_angles']):
            if cls == 'Sign':
                continue
            color = 'red' if cls == 'Vehicle' else 'green' if cls == 'Pedestrian' else 'blue'
            tx, ty, tz = cent[0], cent[1], cent[2]
            c, s = np.cos(yaw), np.sin(yaw)
            sl, sw, sh = dim
            box_to_vehicle = np.array([[sl * c, -sw * s, 0, tx], [sl * s, sw * c, 0, ty], [0, 0, sh, tz], [0, 0, 0, 1]])
            K_4x4 = np.eye(4)
            K_4x4[:3, :3] = K
            R, t = extr
            T_ext = np.eye(4)
            T_ext[:3, :3] = R
            T_ext[:3, 3] = t
            vehicle_to_image = K_4x4 @ T_ext
            vertices = get_3d_box_projected_corners(vehicle_to_image, box_to_vehicle)
            for k in [0, 1]:
                for l in [0, 1]:
                    for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                        x1, y1 = vertices[idx1]
                        x2, y2 = vertices[idx2]
                        seg = clip_line_to_rect(x1, y1, x2, y2, 0, 0, w - 1, h - 1)
                        if seg is not None:
                            ax.plot([seg[0], seg[2]], [seg[1], seg[3]], color=color, linewidth=2)

    def _clear_axes(self):
        for ax in self.axs:
            ax.clear()
            ax.axis("off")

    def show_index(self, idx: int, draw_bbox: bool = True):
        """Render all 18 panels for a given frame index.

        Args:
            idx: Frame index to display.
            draw_bbox: If True, overlay 3D bounding boxes on Livox panels.
        """
        self.idx = max(0, min(idx, self.num_frames - 1))
        self._clear_axes()
        imgs = self._load_images_for_index(self.idx)
        annos = self.data[self.idx].get('annos', {})

        sizes = {cam: (img.shape[1], img.shape[0]) if img is not None else (0, 0) for cam, img in imgs.items()}

        for ax_i, (cam, sensor) in enumerate(self.GRID_ORDER):
            img = imgs.get(cam)
            ax = self.axs[ax_i]
            if img is None:
                w, h = sizes.get("RGB_L", (2048, 1536))
                img = np.zeros((h, w, 3), dtype=np.uint8)
            h, w, _ = img.shape
            is_event = cam.startswith("Event")
            ax.imshow(img, alpha=0.7 if is_event else 1.0)
            pc = self._load_pointcloud_for_sensor(sensor, self.idx)
            if pc is not None:
                xyz, color_vals = pc
                R, t = self.extr[sensor][cam]
                K = self.K[cam]
                d = self.dist[cam]
                uv, c, _ = project_points_to_image(xyz, color_vals, R, t, K, d, w, h, z_min=1.0)
                if len(uv) > 0:
                    point_alpha = 0.4 if sensor == 'Livox' else 0.5
                    point_alpha = 0.7 if cam in ('Event_L', 'Event_R') else point_alpha
                    point_size = 1.0 if sensor == 'Livox' else 2.0
                    ax.scatter(uv[:, 0], uv[:, 1], c=c, cmap='jet', s=point_size, alpha=point_alpha)
            if draw_bbox and sensor == 'Livox' and annos:
                self._draw_3d_bbox(ax, annos, self.K[cam], self.extr[sensor][cam], w, h)
            cam_display = {
                "RGB_L": "Left RGB", "RGB_R": "Right RGB",
                "Event_L": "Left Event", "Event_R": "Right Event",
                "Thermal_L": "Left Thermal", "Thermal_R": "Right Thermal",
            }.get(cam, cam)
            sensor_display = {
                "Livox": "Long-range LiDAR",
                "Ouster": "Short-range LiDAR",
                "Radar": "4D Radar",
            }.get(sensor, sensor)
            ax.set_title(f"{cam_display} + {sensor_display}", fontsize=11, pad=4)

        self.fig.suptitle(f"Frame {self.idx+1}/{self.num_frames}", fontsize=14, fontweight='bold')
        plt.draw()

    def save_jpgs(self, output_dir: str, draw_bbox: bool = True, num_frames: int = None, dpi: int = 80):
        """Save each frame as a JPG file named frame_XXXX.jpg.

        Args:
            output_dir: Directory to save images into (created if not exists).
            draw_bbox: Whether to draw 3D bounding boxes.
            num_frames: Number of frames to render (default: all).
            dpi: Output image resolution.
        """
        ensure_dir(output_dir)
        total = min(num_frames, self.num_frames) if num_frames is not None else self.num_frames
        for i, idx in enumerate(range(total)):
            self.show_index(idx, draw_bbox=draw_bbox)
            out = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            self.fig.savefig(out, dpi=dpi, bbox_inches='tight')
            print(f"Saved {i+1}/{total}: {out}")
        plt.close(self.fig)

# -----------------------------
# Main
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-sensor projection visualizer — saves JPG frames.")
    parser.add_argument("--root_path", type=str, required=True,
                        help="Root dataset path containing session folders")
    parser.add_argument("--bbox", action="store_true",
                        help="Draw 3D bounding boxes on Livox panels")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Max frames to render per session (default: all frames)")
    parser.add_argument("--dpi", type=int, default=80,
                        help="DPI for saved JPG images (default: 80)")
    parser.add_argument("--out_subdir", type=str, default="viz_frames",
                        help="Subdirectory name inside each session folder to save JPGs (default: viz_frames)")
    parser.add_argument("--session", type=str, default=None,
                        help="Process only this session name (e.g. '2024_0101_001'). Searches across all path subdirs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    root_path = args.root_path
    paths = sorted(os.listdir(root_path))
    for path in paths:
        sessions = sorted(os.listdir(os.path.join(root_path, path)))
        for sess in sessions:
            if args.session is not None and sess != args.session:
                continue
            folder_path = os.path.join(root_path, path, sess)
            if not os.path.isdir(folder_path):
                continue
            print(f"Processing: {sess}")
            viewer = MultiSensorProjectionViewer(folder_path, root_path)
            if viewer.pkl_exist:
                out_dir = os.path.join(folder_path, args.out_subdir)
                viewer.save_jpgs(out_dir, draw_bbox=args.bbox, num_frames=args.num_frames, dpi=args.dpi)
            else:
                print(f"No pkl exist: {viewer.pkl_path}")
