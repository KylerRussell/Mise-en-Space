"""Is this footage good enough to map? Fast per-frame and per-stretch checks, with advice.

Structure from motion needs each surface seen from more than one *place*. The checks here
catch what broke our first walkthrough, cheaply enough to run while you film:

- blurry:        motion blur or a dark room; few usable features.
- low_texture:   plain wall or floor filling the view; nothing to track.
- close_up:      one surface fills the view (a shelf, a counter top); depth can't be told apart.
- turning:       the phone turns but doesn't move: every view is from one spot, so nothing
                 beyond arm's reach gets a distance (the bedroom and closet in our first tour).
- lost:          consecutive views share almost nothing (too fast, or a cut).

Motion is measured between frames ~1 s apart: match features, then ask whether a pure
rotation explains the matches (a homography of the form K R K^-1). If it does, the phone only
turned. Otherwise the essential matrix gives the rotation, and the angle left between the rays
once it is undone is the parallax: what lets the solver measure distance.

On our first walkthrough these checks flagged the stretches that later failed to map (blur,
plain walls and jumps in the bedroom, closet and bathroom), but they also flag stretches that
mapped fine: treat them as coaching, and the solve itself as the final word.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

ADVICE = {
    "blurry": "Blurred frames: move more slowly (especially when turning) or turn on more lights.",
    "low_texture": "Little detail in view: keep a corner, the floor edge or furniture in frame instead of a plain wall.",
    "close_up": "Too close to one surface: step back so the room's corners and the floor are in view.",
    "turning": "You turned on the spot here. Walk sideways along the walls instead, about a step at a time, "
               "so each wall is seen from several places.",
    "lost": "The view jumped: slow down and keep part of the previous view in frame.",
    "dark": "Too dark: turn on the lights or open the blinds.",
}


@dataclass
class FrameQC:
    t: float
    sharpness: float
    brightness: float
    features: int
    # filled for frames that have a partner ~window_s later
    matches: int = -1
    rotation_deg: float = float("nan")
    parallax_deg: float = float("nan")
    planar: float = float("nan")  # homography inliers / essential inliers (1 = one plane or pure rotation)
    rotation_only: bool = False  # a pure rotation explains the motion
    flags: list[str] = field(default_factory=list)


@dataclass
class Issue:
    kind: str
    t0: float
    t1: float
    advice: str
    severity: str  # "warn" (map may be weak) or "fail" (this part will likely not map)
    frame_t: float  # a representative moment to show

    def to_dict(self) -> dict:
        return asdict(self)


def default_K(width: int, height: int, fov_deg: float = 62.0) -> np.ndarray:
    """A phone main camera's intrinsics, roughly (62 degrees horizontal). Pass the solved K when known."""
    f = width / 2 / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def _angle(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _parallax(p1: np.ndarray, p2: np.ndarray, Ki: np.ndarray, R: np.ndarray) -> float:
    """Median angle between matching rays once the rotation is undone (x2 ~ R x1 + t)."""
    if len(p1) == 0:
        return float("nan")
    b1 = np.c_[p1, np.ones(len(p1))] @ Ki.T
    b2 = np.c_[p2, np.ones(len(p2))] @ Ki.T
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((b1 * (b2 @ R)).sum(1), -1, 1)))
    return float(np.median(ang))


def measure_motion(p1: np.ndarray, p2: np.ndarray, K: np.ndarray) -> tuple[float, float, float, bool]:
    """(rotation deg, parallax deg, planar ratio, rotation only) between two views from matched pixels.

    nan where unknown. "Rotation only" means a pure rotation explains the matches: no baseline.
    """
    import cv2

    nan = float("nan")
    p1, p2 = np.float32(p1), np.float32(p2)
    E, inl = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or inl is None or E.shape != (3, 3):
        return nan, nan, nan, False
    H, hin = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    planar = float(hin.sum() / max(int(inl.sum()), 1)) if hin is not None else nan
    Ki = np.linalg.inv(K)
    # Pure rotation is the case the essential matrix can't handle (no baseline: its R is noise).
    # It shows up as a homography that is a rotation: K^-1 H K has three equal singular values.
    if H is not None and planar > 0.8:
        U, sv, Vt = np.linalg.svd(Ki @ H @ K)
        if sv[2] / sv[0] > 0.97:
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R = -R
            pts = hin.ravel().astype(bool)
            return _angle(R), _parallax(p1[pts], p2[pts], Ki, R), planar, True
    n_ok, R, _, pin = cv2.recoverPose(E, p1, p2, K, mask=inl.copy())
    if n_ok < 20:
        return nan, nan, planar, False
    pts = pin.ravel().astype(bool)
    return _angle(R), _parallax(p1[pts], p2[pts], Ki, R), planar, False


class Analyzer:
    """Feed frames in time order (BGR images); read back per-frame metrics and issues.

    Keeps only a small window of features, so it works on a live stream from a phone.
    """

    def __init__(self, K: np.ndarray | None = None, window_s: float = 1.0, width: int = 640):
        import cv2

        self.cv2 = cv2
        self.K0 = K
        self.width = width
        self.window_s = window_s
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.hist: list[tuple[float, list, np.ndarray]] = []  # (t, keypoints, descriptors) in the window
        self.frames: list[FrameQC] = []
        self._sharp_ref: list[float] = []

    def _K(self, w: int, h: int) -> np.ndarray:
        if self.K0 is None:
            return default_K(w, h)
        K = self.K0.copy().astype(float)
        s = w / (2 * K[0, 2])  # K given for another resolution: scale it
        K[:2] *= s
        return K

    def add(self, t: float, image: np.ndarray) -> FrameQC:
        cv2 = self.cv2
        h0, w0 = image.shape[:2]
        if w0 != self.width:
            image = cv2.resize(image, (self.width, round(h0 * self.width / w0)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        kp, des = self.orb.detectAndCompute(gray, None)
        fq = FrameQC(t=t, sharpness=sharp, brightness=float(gray.mean()), features=len(kp))
        self._sharp_ref.append(sharp)
        # partner: the frame closest to window_s ago
        while self.hist and t - self.hist[0][0] > self.window_s * 1.6:
            self.hist.pop(0)
        if self.hist and des is not None:
            tp, kpp, desp = min(self.hist, key=lambda h: abs((t - h[0]) - self.window_s))
            if t - tp >= self.window_s * 0.5 and desp is not None:
                self._pair(fq, kpp, desp, kp, des, gray.shape)
        self.hist.append((t, kp, des))
        self._flag(fq)
        self.frames.append(fq)
        return fq

    def _pair(self, fq: FrameQC, kp1, d1, kp2, d2, shape) -> None:
        m = self.matcher.match(d1, d2)
        fq.matches = len(m)
        if len(m) < 25:
            return
        p1 = np.float32([kp1[x.queryIdx].pt for x in m])
        p2 = np.float32([kp2[x.trainIdx].pt for x in m])
        fq.rotation_deg, fq.parallax_deg, fq.planar, fq.rotation_only = measure_motion(p1, p2, self._K(shape[1], shape[0]))

    def _flag(self, fq: FrameQC) -> None:
        ref = np.median(self._sharp_ref[-60:]) if self._sharp_ref else fq.sharpness
        f = fq.flags
        if fq.brightness < 35:
            f.append("dark")
        if fq.sharpness < 0.35 * ref or fq.sharpness < 15:
            f.append("blurry")
        if fq.features < 250:
            f.append("low_texture")
        if fq.matches >= 0 and fq.matches < 40:
            f.append("lost")
        if not np.isnan(fq.parallax_deg):
            if fq.rotation_only and fq.rotation_deg > 6:
                f.append("turning")
            elif fq.planar > 0.9 and fq.parallax_deg > 2.0 and fq.features < 700:
                f.append("close_up")

    def issues(self, min_len_s: float = 1.5, merge_gap_s: float = 1.0) -> list[Issue]:
        """Runs of a flag at least min_len_s long, merged across short gaps; most severe first."""
        out = []
        for kind, advice in ADVICE.items():
            ts = [fq.t for fq in self.frames if kind in fq.flags]
            runs: list[list[float]] = []
            for t in ts:
                if runs and t - runs[-1][1] <= merge_gap_s + 1e-6:
                    runs[-1][1] = t
                else:
                    runs.append([t, t])
            for t0, t1 in runs:
                if t1 - t0 + 1e-6 >= min_len_s:
                    sev = "fail" if kind in ("turning", "lost") or t1 - t0 > 6 else "warn"
                    out.append(Issue(kind, round(t0, 2), round(t1, 2), advice, sev, round((t0 + t1) / 2, 2)))
        out.sort(key=lambda i: (i.severity != "fail", -(i.t1 - i.t0)))
        return out

    def timeline(self) -> list[dict]:
        """Per frame: time and the worst flag (or 'ok'), for drawing a status strip."""
        order = ["lost", "turning", "close_up", "blurry", "low_texture", "dark"]
        out = []
        for fq in self.frames:
            worst = next((k for k in order if k in fq.flags), "ok")
            out.append({"t": round(fq.t, 2), "status": worst,
                        "parallax": None if np.isnan(fq.parallax_deg) else round(fq.parallax_deg, 2),
                        "rotation": None if np.isnan(fq.rotation_deg) else round(fq.rotation_deg, 1)})
        return out


def analyze_frames(frames: list[tuple[float, np.ndarray]], K: np.ndarray | None = None) -> Analyzer:
    a = Analyzer(K)
    for t, img in frames:
        a.add(t, img)
    return a


def analyze_video(path, fps: float = 3.0, K: np.ndarray | None = None, on_frame=None) -> Analyzer:
    """Decode at ~fps and analyze. on_frame(FrameQC, image) is called as frames go by (for live progress)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    a = Analyzer(K)
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok:
                fq = a.add(i / src_fps, img)
                if on_frame:
                    on_frame(fq, img)
        i += 1
    cap.release()
    return a
