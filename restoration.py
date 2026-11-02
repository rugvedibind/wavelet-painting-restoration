"""Wavelet transform based restoration of a damaged painting.

The same pipeline as notebooks/DIP_Wavelet_Restoration_Chinese_Painting.ipynb, packaged
as plain functions so it can be reused by the Streamlit app (app.py) or other scripts.

Stages: 3.1 pre-processing, 3.2 damage detection, 3.3 segmentation, 3.4 DWT,
3.5 BayesShrink thresholding, 3.6 inverse DWT, 3.7 inpainting, 3.8 post-processing.
"""
import cv2
import numpy as np
import pywt
from skimage.metrics import structural_similarity


# ----------------------------------------------------------------- helpers
def to_f(img):
    return img.astype(np.float32) / 255.0


def to_u8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def prepare(img_rgb, max_side=768, multiple=16):
    """Resize so the longer side is max_side and both sides are divisible by `multiple`."""
    h, w = img_rgb.shape[:2]
    s = max_side / max(h, w)
    img = cv2.resize(img_rgb, (int(round(w * s)), int(round(h * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    return img[: h - h % multiple, : w - w % multiple].copy()


def evaluate(ref, img, region=None):
    """MSE (0-255 scale), PSNR (dB) and SSIM against a reference; PSNR only if `region` is given."""
    a = to_u8(ref).astype(np.float64)
    b = to_u8(img).astype(np.float64)
    if region is not None:
        if region.sum() == 0:
            return float("nan")
        mse = np.mean((a[region] - b[region]) ** 2)
        return float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))
    mse = float(np.mean((a - b) ** 2))
    psnr = float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=255))
    return mse, psnr, ssim


# ----------------------------------------------------------------- damage simulation
def _blob_mask(rng, h, w, coverage, scale=10):
    coarse = rng.random((max(h // scale, 4), max(w // scale, 4))).astype(np.float32)
    field = cv2.GaussianBlur(cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 18)
    field = (field - field.min()) / (np.ptp(field) + 1e-8)
    fine = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    field = field + 0.06 * (fine - 0.5)
    return field > np.quantile(field, 1 - coverage)


def _random_walks(rng, h, w, n, thickness, step, length, straight=False):
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, 2 * np.pi)
        pts = [(x, y)]
        for _ in range(length):
            ang += rng.normal(0, 0.05 if straight else 0.35)
            x += step * np.cos(ang)
            y += step * np.sin(ang)
            pts.append((x, y))
        cv2.polylines(m, [np.int32(pts)], False, 1, thickness, cv2.LINE_8)
    return m.astype(bool)


def simulate_damage(img, seed=42, coverage=0.10, n_tears=4, n_scratches=6, noise_sigma=0.05, fading=0.0):
    """Return (damaged image, ground-truth damage mask). Same random sequence as the notebook."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    holes = _blob_mask(rng, h, w, coverage) if coverage > 0 else np.zeros((h, w), bool)
    tears = _random_walks(rng, h, w, n_tears, 4, 6, 45)
    scratch = _random_walks(rng, h, w, n_scratches, 2, 8, 30, straight=True)
    gt = holes | tears | scratch
    dmg = img.copy()
    if fading > 0:
        dmg = (1 - fading) * dmg + fading * np.array([0.80, 0.72, 0.55], np.float32)
    dmg[gt] = np.array([0.965, 0.955, 0.93], np.float32)
    dmg = dmg + rng.normal(0, noise_sigma, dmg.shape).astype(np.float32)
    return np.clip(dmg, 0, 1).astype(np.float32), gt


# ----------------------------------------------------------------- 3.1 - 3.3 detection
def preprocess(img):
    u8 = to_u8(img)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(u8, (0, 0), 1.2)
    median = cv2.medianBlur(u8, 3)
    return gray, smooth, median


def _keep_large(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def detect_damage(smooth, median, v_thr=225, s_thr=40, tophat_thr=28):
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    regions = (hsv[..., 2] >= v_thr) & (hsv[..., 1] <= s_thr)
    g = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    hsv_m = cv2.cvtColor(median, cv2.COLOR_RGB2HSV)
    lines = (tophat > tophat_thr) & (hsv_m[..., 2] > 200) & (hsv_m[..., 1] < 70)
    lines = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lines = _keep_large(lines, 15)
    return regions, lines, tophat


def segment(raw, min_area=30, margin=2):
    m = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = _keep_large(m, min_area).astype(np.uint8)
    if margin > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
    n, _, _, _ = cv2.connectedComponentsWithStats(m, 8)
    return m.astype(bool), n - 1


def detection_scores(mask, gt):
    tp = float((mask & gt).sum()); fp = float((mask & ~gt).sum()); fn = float((~mask & gt).sum())
    return {"Precision": tp / max(tp + fp, 1), "Recall": tp / max(tp + fn, 1),
            "IoU": tp / max(tp + fp + fn, 1), "Dice": 2 * tp / max(2 * tp + fp + fn, 1)}


# ----------------------------------------------------------------- 3.4 - 3.6 wavelet restoration
def estimate_sigma(ch, dmask, wavelet="db4"):
    """Robust noise estimate: median(|HH1|) / 0.6745, ignoring damaged pixels."""
    _, (_, _, hh) = pywt.dwt2(ch, wavelet, mode="periodization")
    mm = cv2.resize(dmask.astype(np.uint8), (hh.shape[1], hh.shape[0]), interpolation=cv2.INTER_NEAREST)
    mm = cv2.dilate(mm, np.ones((5, 5), np.uint8)).astype(bool)
    vals = hh[~mm] if (~mm).sum() > 100 else hh.ravel()
    return float(np.median(np.abs(vals)) / 0.6745)


def threshold_value(c, sigma, rule, n_pixels):
    if rule == "visu":
        return sigma * np.sqrt(2 * np.log(n_pixels))
    sx = np.sqrt(max(np.mean(c ** 2) - sigma ** 2, 1e-12))
    return sigma ** 2 / sx


def wavelet_threshold(coeffs, sigma, n_pixels, mode="soft", rule="bayes", gain=1.0, log=None):
    new = [coeffs[0]]
    for i, lv in enumerate(coeffs[1:]):
        level = len(coeffs) - 1 - i
        bands = []
        for name, c in zip(["LH", "HL", "HH"], lv):
            t = threshold_value(c, sigma, rule, n_pixels)
            c2 = pywt.threshold(c, t, mode) * gain
            if log is not None:
                log.append({"Level": level, "Sub-band": name, "Threshold": t, "Zeroed (%)": 100 * float(np.mean(c2 == 0))})
            bands.append(c2)
        new.append(tuple(bands))
    return new


def wavelet_denoise(img, dmask, wavelet="db4", level=3, mode="soft", rule="bayes", gain=1.0, log=None):
    """3.4 DWT -> 3.5 thresholding -> 3.6 inverse DWT, per RGB channel."""
    h, w = img.shape[:2]
    out = np.empty_like(img)
    sigmas = []
    for c in range(3):
        s = estimate_sigma(img[..., c], dmask, wavelet)
        sigmas.append(s)
        cs = pywt.wavedec2(img[..., c], wavelet, level=level, mode="periodization")
        new = wavelet_threshold(cs, s, h * w, mode, rule, gain, log=log if c == 1 else None)
        out[..., c] = pywt.waverec2(new, wavelet, mode="periodization")[:h, :w]
    return np.clip(out, 0, 1).astype(np.float32), sigmas


def subbands(gray_f, wavelet="db4"):
    LL, (LH, HL, HH) = pywt.dwt2(gray_f, wavelet, mode="periodization")
    return {"LL": LL, "LH": LH, "HL": HL, "HH": HH}


def visual_band(c, detail=True):
    if detail:
        c = np.abs(c)
        return np.clip(c / (np.percentile(c, 99.5) + 1e-8), 0, 1)
    return (c - c.min()) / (np.ptp(c) + 1e-8)


# ----------------------------------------------------------------- 3.7 - 3.8 reconstruction
def inpaint(img, dmask, radius=5, method="telea"):
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return to_f(cv2.inpaint(to_u8(img), dmask.astype(np.uint8), radius, flag))


def seam_smooth(img, dmask):
    m = dmask.astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    ring = (cv2.dilate(m, k) - cv2.erode(m, k)).astype(np.float32)
    alpha = np.clip(cv2.GaussianBlur(ring, (0, 0), 2) * 1.5, 0, 1)[..., None]
    smooth = cv2.bilateralFilter(img.astype(np.float32), 7, 0.08, 5)
    return np.clip(alpha * smooth + (1 - alpha) * img, 0, 1).astype(np.float32)


def enhance(img, clip=1.0, amount=0.3):
    """Optional display enhancement: CLAHE on lightness + unsharp mask (lowers PSNR)."""
    lab = cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2LAB)
    if clip > 0:
        lab[..., 0] = cv2.createCLAHE(clip, (8, 8)).apply(lab[..., 0])
    x = to_f(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    return np.clip(x + amount * (x - cv2.GaussianBlur(x, (0, 0), 1.0)), 0, 1).astype(np.float32)


def restore(damaged, v_thr=225, s_thr=40, tophat_thr=28, min_area=30, margin=2,
            wavelet="db4", level=3, mode="soft", rule="bayes", gain=1.0, radius=5, method="telea"):
    """Full pipeline. Returns a dict with every intermediate result."""
    gray, smooth, median = preprocess(damaged)                                  # 3.1
    regions, lines, tophat = detect_damage(smooth, median, v_thr, s_thr, tophat_thr)  # 3.2
    mask, n_regions = segment(regions | lines, min_area, margin)               # 3.3
    log = []
    denoised, sigmas = wavelet_denoise(damaged, mask, wavelet, level, mode, rule, gain, log)  # 3.4-3.6
    reconstructed = inpaint(denoised, mask, radius, method)                    # 3.7
    restored = seam_smooth(reconstructed, mask)                                # 3.8
    return {"gray": gray, "raw_mask": regions | lines, "tophat": tophat, "mask": mask, "n_regions": n_regions,
            "denoised": denoised, "sigmas": sigmas, "thresholds": log,
            "reconstructed": reconstructed, "restored": restored}
