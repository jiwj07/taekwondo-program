# evaluator.py

"""
Taekwondo movement evaluator using MMPose + your parametric templates
- Multiple movements, each with its own joint set and x(t), y(t) functions
- Mirroring (left/right) via reflection about a time-varying anchor (hip/head)
- Optional limb-length retargeting
- Procrustes (similarity) alignment per frame -> translation/scale/rotation invariance
- Optional DTW time alignment (set use_dtw=True in evaluate)

Notes on polynomial typos from the source list (fixed here):
- Arae makki, right elbow y(t): interpreted as
    5.2636 t^5 - 13.4840 t^4 + 12.3260 t^3 - 4.5470 t^2 + 0.5069 t + 1.2876
  (the original had a malformed "- 4.547tt^2" and duplicated t terms)
- Momtong makki, left fist x(t)/y(t): replaced stray 'x' with 't'.
- Momtong makki, right shoulder y(t): "-083.037" -> "-83.037".
Update these if your ground truth differs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import os
import numpy as np
import cv2
from scipy.linalg import svd

try:
    from dtw import dtw  # pip install dtw-python
    _HAS_DTW = True
except Exception:
    _HAS_DTW = False

try:
    from mmpose.apis import MMPoseInferencer
    _HAS_MMPOSE = True
except Exception:
    _HAS_MMPOSE = False

# ------------------------------
# COCO-17 skeleton indices
# ------------------------------
COCO = dict(
    nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4,
    l_shoulder=5, r_shoulder=6, l_elbow=7, r_elbow=8,
    l_wrist=9, r_wrist=10, l_hip=11, r_hip=12,
    l_knee=13, r_knee=14, l_ankle=15, r_ankle=16,
)

ALIAS_TO_INDEX = {
    'L_foot': COCO['l_ankle'],
    'R_foot': COCO['r_ankle'],
    'L_knee': COCO['l_knee'],
    'R_knee': COCO['r_knee'],
    'L_elbow': COCO['l_elbow'],
    'R_elbow': COCO['r_elbow'],
    'L_fist': COCO['l_wrist'],   # wrists as proxies for fists (upgrade to WholeBody for true fists/toes)
    'R_fist': COCO['r_wrist'],
    'L_shoulder': COCO['l_shoulder'],
    'R_shoulder': COCO['r_shoulder'],
    # 'hip' and 'head' are virtual and computed from other points
}

LR_SWAP = {
    'L_foot':'R_foot','R_foot':'L_foot',
    'L_knee':'R_knee','R_knee':'L_knee',
    'L_elbow':'R_elbow','R_elbow':'L_elbow',
    'L_fist':'R_fist','R_fist':'L_fist',
    'L_shoulder':'R_shoulder','R_shoulder':'L_shoulder',
    'hip':'hip','head':'head'
}

# ------------------------------
# Virtual points
# ------------------------------

def pelvis_xy(frame17: np.ndarray) -> np.ndarray:
    return (frame17[COCO['l_hip']] + frame17[COCO['r_hip']]) / 2.0

def head_center_xy(frame17: np.ndarray) -> np.ndarray:
    pts = [COCO['nose'], COCO['l_eye'], COCO['r_eye'], COCO['l_ear'], COCO['r_ear']]
    ok = [p for p in pts if np.isfinite(frame17[p]).all()]
    if not ok:
        return np.array([np.nan, np.nan])
    return frame17[ok].mean(axis=0)

# ------------------------------
# Movement spec / registry
# ------------------------------
@dataclass
class MovementSpec:
    name: str
    aliases: List[str]
    funcs: Dict[str, Callable[[float], np.ndarray]]  # alias -> f(t)->(x,y) (t in [0,1])
    anchor_alias: str = 'hip'  # reflection anchor
    retarget_edges: List[Tuple[str,str]] = field(default_factory=list)  # optional bone retargeting

# ——— Movement function sets (RIGHT/LEFT per your list) ———

def right_ap_seogi_funcs() -> Dict[str, Callable[[float], np.ndarray]]:
    def L_foot(t): return np.array([0.59, 0.14])
    def L_knee(t): return np.array([0.2074*t + 0.4847,
                                    0.0733*t**4 - 0.1989*t**3 + 0.129*t**2 + 0.0071*t + 0.546])
    def hip(t):    return np.array([0.3359*t + 0.3787,
                                    0.0560*t**4 - 0.1954*t**3 + 0.1971*t**2 - 0.0596*t + 0.9914])
    def R_knee(t):return np.array([0.5870*t + 0.1867,
                                    -0.0994*t**2 + 0.1406*t + 0.5300])
    def R_foot(t):return np.array([0.8438*t - 0.0786,
                                    0.3661*t**4 - 1.0228*t**3 + 0.7254*t**2 - 0.0460*t + 0.1390])
    return {'L_foot':L_foot,'L_knee':L_knee,'hip':hip,'R_knee':R_knee,'R_foot':R_foot}


def right_ap_chagi_funcs() -> Dict[str, Callable[[float], np.ndarray]]:
    def L_foot(t): return np.array([0.6900, 0.1300])
    def L_knee(t): return np.array([0.0863*t + 0.6558,
                                    -0.3704*t**5 + 1.8446*t**4 - 3.1465*t**3 + 2.0026*t**2 + 0.4830])
    def hip(t):    return np.array([0.2218*t + 0.5317,
                                    -0.1122*t**2 + 0.1864*t + 0.8955])
    def R_knee(t):return np.array([-0.5926*t**2 + 1.4319*t + 0.3078,
                                    1.1101*t**3 - 3.9583*t**2 + 3.7593*t + 0.1096])
    def R_foot(t):return np.array([-0.8661*t**2 + 2.1206*t + 0.0426,
                                    1.7849*t**3 - 6.2892*t**2 + 5.8082*t - 0.4987])
    def head(t):  return np.array([0.2870*t + 0.4466,
                                    -0.0313*t**2 + 0.0795*t + 1.6666])
    return {'L_foot':L_foot,'L_knee':L_knee,'hip':hip,'R_knee':R_knee,'R_foot':R_foot,'head':head}


def right_ap_gubi_funcs() -> Dict[str, Callable[[float], np.ndarray]]:
    def L_foot(t): return np.array([0.9300, 0.1200])
    def L_knee(t): return np.array([0.1326*t + 0.9841,
                                    0.0227*t**2 - 0.0759*t + 0.5106])
    def hip(t):    return np.array([
        -0.1191*t**5 + 0.7457*t**4 - 1.6581*t**3 + 1.5200*t**2 - 0.4806*t + 1.5172,
         0.4516*t + 0.7617
    ])
    def R_knee(t):return np.array([0.7744*t + 0.5066,
                                    -0.0237*t**3 + 0.0590*t**2 + 0.0124*t + 0.4620])
    def R_foot(t):return np.array([
         0.9540*t + 0.1779,
         0.2929*t**6 - 2.1359*t**5 + 5.8894*t**4 - 7.5048*t**3 + 4.2331*t**2 - 0.7730*t + 0.1141
    ])
    return {'L_foot':L_foot,'L_knee':L_knee,'hip':hip,'R_knee':R_knee,'R_foot':R_foot}


def left_arae_makki_funcs() -> Dict[str, Callable[[float], np.ndarray]]:
    def L_fist(t):   return np.array([0.8741*t + 0.1474,
                                      -0.8934*t + 1.6191])
    def L_elbow(t):  return np.array([0.1786*t + 0.4633,
                                      -0.3742*t + 1.4563])
    def head(t):     return np.array([0.3800, 1.7000])
    def R_elbow(t):  return np.array([-0.4626*t + 0.3520,
                                      5.2636*t**5 - 13.4840*t**4 + 12.3260*t**3 - 4.5470*t**2 + 0.5069*t + 1.2876])
    def R_fist(t):   return np.array([-0.5902*t + 0.7129,
                                      -0.6464*t**3 + 0.8851*t**2 - 0.2567*t + 1.1724])
    return {'L_fist':L_fist,'L_elbow':L_elbow,'head':head,'R_elbow':R_elbow,'R_fist':R_fist}


def right_momtong_makki_funcs() -> Dict[str, Callable[[float], np.ndarray]]:
    def L_fist(t):      return np.array([-1.5982*t + 1.3785, -0.2990*t + 1.2415])
    def L_elbow(t):     return np.array([-1.4961*t + 1.1486,
                                         -15.9860*t**4 + 20.8000*t**3 - 8.0494*t**2 + 0.7850*t + 1.2542])
    def L_shoulder(t):  return np.array([-0.7286*t + 0.8559,
                                         -25.0090*t**5 + 38.0250*t**4 - 19.0200*t**3 + 3.3308*t**2 - 0.1087*t + 1.3942])
    def head(t):        return np.array([0.6800, 1.6300])
    def R_shoulder(t):  return np.array([0.2333*t + 0.5251,
                                         -83.0370*t**6 + 149.10*t**5 - 100.31*t**4 + 31.635*t**3 - 4.918*t**2 + 0.4070*t + 1.3910])
    def R_elbow(t):     return np.array([1.1083*t + 0.3728, -0.1430*t + 1.2878])
    def R_fist(t):      return np.array([1.8585*t + 0.2460, -0.4328*t + 1.6017])
    return {
        'L_fist':L_fist,'L_elbow':L_elbow,'L_shoulder':L_shoulder,
        'head':head,'R_shoulder':R_shoulder,'R_elbow':R_elbow,'R_fist':R_fist
    }


def left_olgul_makki_funcs() -> Dict[str, Callable[[float], np.ndarray]]:
    def L_fist(t):   return np.array([0.6006*t**3 - 1.0520*t**2 + 0.5846*t + 0.4677,
                                      -2.1389*t**2 + 2.5386*t + 1.1930])
    def L_elbow(t):  return np.array([-3.2700*t**4 + 5.9761*t**3 - 3.1766*t**2 + 0.3222*t + 0.5386,
                                      -1.5328*t**2 + 1.7366*t + 1.2062])
    def head(t):     return np.array([0.3800, 1.6400])
    def R_elbow(t):  return np.array([0.9346*t**2 - 1.1575*t + 0.4457,
                                      -1.5860*t**3 + 2.5606*t**2 - 1.2124*t + 1.3623])
    def R_fist(t):   return np.array([0.5659*t**2 - 0.7286*t + 0.5776,
                                      1.0934*t**2 - 1.3470*t + 1.4880])
    return {'L_fist':L_fist,'L_elbow':L_elbow,'head':head,'R_elbow':R_elbow,'R_fist':R_fist}

# ------------------------------
# Registry of movements
# ------------------------------
MOVEMENTS: Dict[str, MovementSpec] = {
    'right_ap_seogi': MovementSpec(
        name='right_ap_seogi',
        aliases=['L_foot','L_knee','hip','R_knee','R_foot'],
        funcs=right_ap_seogi_funcs(),
        anchor_alias='hip',
        retarget_edges=[('hip','L_knee'),('L_knee','L_foot'),('hip','R_knee'),('R_knee','R_foot')]
    ),
    'right_ap_chagi': MovementSpec(
        name='right_ap_chagi',
        aliases=['L_foot','L_knee','hip','R_knee','R_foot','head'],
        funcs=right_ap_chagi_funcs(),
        anchor_alias='hip',
        retarget_edges=[('hip','L_knee'),('L_knee','L_foot'),('hip','R_knee'),('R_knee','R_foot')]
    ),
    'right_ap_gubi': MovementSpec(
        name='right_ap_gubi',
        aliases=['L_foot','L_knee','hip','R_knee','R_foot'],
        funcs=right_ap_gubi_funcs(),
        anchor_alias='hip',
        retarget_edges=[('hip','L_knee'),('L_knee','L_foot'),('hip','R_knee'),('R_knee','R_foot')]
    ),
    'left_arae_makki': MovementSpec(
        name='left_arae_makki',
        aliases=['L_fist','L_elbow','head','R_elbow','R_fist'],
        funcs=left_arae_makki_funcs(),
        anchor_alias='head',
        retarget_edges=[]
    ),
    'right_momtong_makki': MovementSpec(
        name='right_momtong_makki',
        aliases=['L_fist','L_elbow','L_shoulder','head','R_shoulder','R_elbow','R_fist'],
        funcs=right_momtong_makki_funcs(),
        anchor_alias='head',
        retarget_edges=[('L_shoulder','L_elbow'),('L_elbow','L_fist'),('R_shoulder','R_elbow'),('R_elbow','R_fist')]
    ),
    'left_olgul_makki': MovementSpec(
        name='left_olgul_makki',
        aliases=['L_fist','L_elbow','head','R_elbow','R_fist'],
        funcs=left_olgul_makki_funcs(),
        anchor_alias='head',
        retarget_edges=[]
    ),
}

# ------------------------------
# Template utilities
# ------------------------------

def sample_template(spec: MovementSpec, times_01: np.ndarray) -> np.ndarray:
    out = np.zeros((len(times_01), len(spec.aliases), 2), float)
    for i, tt in enumerate(times_01):
        for j, a in enumerate(spec.aliases):
            out[i, j, :] = spec.funcs[a](tt)
    return out


def reflect_template(spec: MovementSpec, X: np.ndarray, times_01: np.ndarray) -> Tuple[List[str], np.ndarray]:
    anchor = np.array([spec.funcs[spec.anchor_alias](tt) for tt in times_01])  # (T,2)
    Xmir = X.copy()
    Xmir[..., 0] = 2*anchor[:, None, 0] - Xmir[..., 0]
    aliases_m = [LR_SWAP.get(a, a) for a in spec.aliases]
    # reorder to align names after swap
    order = [aliases_m.index(a) if a in aliases_m else i for i, a in enumerate(spec.aliases)]
    Xmir = Xmir[:, order, :]
    return aliases_m, Xmir


def retarget_limb_lengths(spec: MovementSpec, tpl: np.ndarray, obs: np.ndarray) -> np.ndarray:
    if not spec.retarget_edges:
        return tpl
    out = tpl.copy()
    alias_idx = {a:i for i,a in enumerate(spec.aliases)}
    for t in range(tpl.shape[0]):
        for parent, child in spec.retarget_edges:
            if parent not in alias_idx or child not in alias_idx:
                continue
            pi, ci = alias_idx[parent], alias_idx[child]
            v_tpl = out[t, ci] - out[t, pi]
            v_obs = obs[t, ci] - obs[t, pi]
            len_tpl = float(np.linalg.norm(v_tpl)) + 1e-8
            len_obs = float(np.linalg.norm(v_obs))
            out[t, ci] = out[t, pi] + v_tpl * (len_obs / len_tpl)
    return out

# ------------------------------
# Pose extraction (MMPose)
# ------------------------------

def _read_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def extract_alias_series(video_path: str, kpt_thr: float, aliases: List[str], model_alias: str='human') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert _HAS_MMPOSE, "MMPose not available. Please install mmpose / mmengine / mmcv."
    infer = MMPoseInferencer(model_alias)
    fps = _read_fps(video_path)
    times: List[float] = []
    frames: List[np.ndarray] = []
    confs: List[np.ndarray] = []

    for i, out in enumerate(infer(video_path, return_vis=False, kpt_thr=kpt_thr, draw_bbox=False, use_oks_tracking=True)):
        preds = out['predictions'][0]
        if not preds:
            continue
        inst = preds[0]  # single-person assumption; adjust if needed
        k = np.array(inst['keypoints'][..., :2], float)   # (17,2)
        s = np.array(inst['keypoint_scores'], float)      # (17,)
        J = len(aliases)
        P = np.zeros((J,2), float); P[:] = np.nan
        W = np.zeros((J,), float)
        for j, a in enumerate(aliases):
            if a == 'hip':
                xy = pelvis_xy(k)
                conf = min(s[COCO['l_hip']], s[COCO['r_hip']])
            elif a == 'head':
                xy = head_center_xy(k)
                conf = float(np.isfinite(xy).all())
            else:
                idx = ALIAS_TO_INDEX[a]
                xy = k[idx]
                conf = s[idx]
            if not np.isfinite(xy).all() or conf < kpt_thr:
                pass
            else:
                P[j] = xy; W[j] = conf
        times.append(i / fps)
        frames.append(P)
        confs.append(W)

    T = len(times)
    if T == 0:
        raise RuntimeError("No frames extracted from video.")

    Pseq = np.stack(frames, 0)  # (T,J,2)
    Wseq = np.stack(confs, 0)   # (T,J)
    # Fill short gaps for stability
    for j in range(Pseq.shape[1]):
        col = Pseq[:, j, :]
        mask = np.isfinite(col[:, 0])
        if mask.any():
            for t in range(1, T):
                if not mask[t]: col[t] = col[t-1]
            first = int(np.argmax(mask))
            for t in range(first): col[t] = col[first]
            Pseq[:, j, :] = col
            Wseq[:, j] = (Wseq[:, j] > 0).astype(float)
    return np.array(times), Pseq, Wseq

# ------------------------------
# Alignment & scoring
# ------------------------------

def normalize_times_to_01(times: np.ndarray) -> np.ndarray:
    if len(times) <= 1:
        return np.zeros_like(times)
    return (times - times[0]) / (times[-1] - times[0] + 1e-8)


def dtw_align_template_to_obs(tpl_dense: np.ndarray, obs_seq: np.ndarray) -> np.ndarray:
    if not _HAS_DTW:
        idx = np.linspace(0, len(tpl_dense)-1, obs_seq.shape[0]).round().astype(int)
        return tpl_dense[idx]
    A = obs_seq.reshape(obs_seq.shape[0], -1)
    B = tpl_dense.reshape(tpl_dense.shape[0], -1)
    al = dtw(A, B, keep_internals=False)
    i_idx, j_idx = np.array(al.index1), np.array(al.index2)
    out = np.zeros_like(obs_seq)
    last = 0
    for i in range(obs_seq.shape[0]):
        js = np.where(i_idx == i)[0]
        if len(js) > 0:
            last = int(j_idx[js].max())
        out[i] = tpl_dense[last]
    return out


def weighted_similarity_transform(A: np.ndarray, B: np.ndarray, w: np.ndarray):
    w = w / (w.sum() + 1e-8)
    muA = (w[:, None] * A).sum(0)
    muB = (w[:, None] * B).sum(0)
    A0, B0 = A - muA, B - muB
    H = (A0 * w[:, None]).T @ B0
    U, S, Vt = svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    s = S.sum() / ((w[:, None] * (A0**2)).sum() + 1e-8)
    t = muB - s * (R @ muA)
    return s, R, t


def frame_error(template_J2: np.ndarray, observed_J2: np.ndarray, conf_J: np.ndarray) -> Tuple[np.ndarray, float]:
    s, R, t = weighted_similarity_transform(template_J2, observed_J2, conf_J)
    mapped = (s * (template_J2 @ R.T)) + t
    d = np.linalg.norm(mapped - observed_J2, axis=1)
    rmse = float(np.sqrt((conf_J * (d**2)).sum() / (conf_J.sum() + 1e-8)))
    return d, rmse

# ------------------------------
# Public API
# ------------------------------

def available_movements() -> List[str]:
    return list(MOVEMENTS.keys())


def evaluate(video_path: str, movement_key: str, side: str='auto', use_dtw: bool=True, kpt_thr: float=0.3, model_alias: str='human') -> Dict:
    assert movement_key in MOVEMENTS, f"Unknown movement '{movement_key}'."
    spec = MOVEMENTS[movement_key]
    times, P_obs, W_obs = extract_alias_series(video_path, kpt_thr, spec.aliases, model_alias=model_alias)
    t01 = normalize_times_to_01(times)
    tpl = sample_template(spec, t01)
    # DTW (optional) — we could align dense→obs if desired. For now, sample directly to t01.
    # Mirror template using time-varying anchor
    aliases_m, tpl_mirror = reflect_template(spec, tpl, t01)
    # Retarget limbs to performer proportions
    tpl_rt  = retarget_limb_lengths(spec, tpl, P_obs)
    tplm_rt = retarget_limb_lengths(spec, tpl_mirror, P_obs)

    def mean_rmse_for(X):
        rmses = [frame_error(X[i], P_obs[i], W_obs[i])[1] for i in range(len(times))]
        return float(np.mean(rmses)), rmses

    rmse_o, per_o = mean_rmse_for(tpl_rt)
    rmse_m, per_m = mean_rmse_for(tplm_rt)
    # side handling: 'right' -> original, 'left' -> mirrored, 'auto' -> pick best
    if side.lower() == 'right':
        chosen = 'original'; per = per_o; mean_rmse = rmse_o
    elif side.lower() == 'left':
        chosen = 'mirrored'; per = per_m; mean_rmse = rmse_m
    else:
        chosen = 'mirrored' if rmse_m < rmse_o else 'original'
        per = per_m if chosen == 'mirrored' else per_o
        mean_rmse = rmse_m if chosen == 'mirrored' else rmse_o
    score = float(np.clip(1.0 - mean_rmse/0.15, 0.0, 1.0))  # adjust tolerance as needed

    return {
        'movement': movement_key,
        'side_requested': side,
        'frames': len(times),
        'orientation_used': chosen,
        'mean_rmse': mean_rmse,
        'overall_score': score,
        'per_frame_rmse': per,
    }

if __name__ == '__main__':
    print('Available movements:')
    for k in available_movements():
        print(' -', k)
