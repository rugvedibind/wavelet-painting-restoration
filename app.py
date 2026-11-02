"""Streamlit app: wavelet transform based restoration of a damaged Chinese painting.

Run locally:  streamlit run app.py
"""
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.restoration import estimate_sigma as sk_sigma

import restoration as R

st.set_page_config(page_title="Wavelet Painting Restoration", page_icon="🖌️", layout="wide")

REFERENCE = Path(__file__).parent / "data" / "reference_painting.png"


# ----------------------------------------------------------------- cached computation
@st.cache_data(show_spinner=False)
def load_image(data: bytes, max_side: int):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    return R.to_f(R.prepare(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), max_side))


@st.cache_data(show_spinner=False)
def run_simulation(img, seed, coverage, tears, scratches, noise, fading):
    return R.simulate_damage(img, seed, coverage, tears, scratches, noise, fading)


@st.cache_data(show_spinner=False)
def run_restore(damaged, params: dict):
    return R.restore(damaged, **params)


@st.cache_data(show_spinner=False)
def run_baselines(original, damaged, gt, mask, denoised, restored, radius):
    rows = {
        "Damaged input": damaged,
        "Telea inpainting (no denoising)": R.inpaint(damaged, mask, radius, "telea"),
        "Navier–Stokes inpainting (no denoising)": R.inpaint(damaged, mask, radius, "ns"),
        "Wavelet denoising only (no inpainting)": denoised,
        "Proposed: wavelet + inpainting": restored,
        "Proposed + display enhancement": R.enhance(restored),
    }
    out = []
    for name, img in rows.items():
        mse, psnr, ssim = R.evaluate(original, img)
        out.append({"Method": name, "MSE": round(mse, 2), "PSNR (dB)": round(psnr, 2), "SSIM": round(ssim, 3),
                    "PSNR damaged area": round(R.evaluate(original, img, gt), 2),
                    "PSNR intact area": round(R.evaluate(original, img, ~gt), 2)})
    return pd.DataFrame(out)


def png_bytes(img):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(R.to_u8(img), cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def overlay(img, mask):
    ov = img.copy()
    ov[mask] = 0.55 * ov[mask] + 0.45 * np.array([0.89, 0.12, 0.12], np.float32)
    return ov


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
mode = st.sidebar.radio("Mode", ["Simulate damage (with metrics)", "Restore my damaged image"],
                        help="Simulation starts from a clean painting so MSE, PSNR and SSIM can be measured.")
simulate = mode.startswith("Simulate")

st.sidebar.subheader("1. Input image")
max_side = st.sidebar.select_slider("Working size (longer side, px)", [384, 512, 640, 768, 1024], value=768)
if simulate:
    source = st.sidebar.radio("Clean painting", ["Reference painting", "Upload my own"], horizontal=True)
    upload = st.sidebar.file_uploader("Clean image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]) if source == "Upload my own" else None
    data = upload.getvalue() if upload else (REFERENCE.read_bytes() if source == "Reference painting" else None)
else:
    upload = st.sidebar.file_uploader("Damaged image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    data = upload.getvalue() if upload else None

if simulate:
    st.sidebar.subheader("2. Simulated damage")
    coverage = st.sidebar.slider("Torn / missing patches (fraction)", 0.0, 0.30, 0.10, 0.01)
    tears = st.sidebar.slider("Cracks / tears", 0, 12, 4)
    scratches = st.sidebar.slider("Scratches", 0, 15, 6)
    noise = st.sidebar.slider("Noise σ", 0.0, 0.15, 0.05, 0.01)
    fading = st.sidebar.slider("Fading / yellowing", 0.0, 0.4, 0.0, 0.05)
    seed = int(st.sidebar.number_input("Random seed", 0, 99999, 42))

st.sidebar.subheader("3. Damage detection")
with st.sidebar.expander("Detection thresholds", expanded=not simulate):
    v_thr = st.slider("Brightness V ≥", 150, 255, 225)
    s_thr = st.slider("Saturation S ≤", 0, 120, 40)
    tophat_thr = st.slider("Scratch (top-hat) >", 5, 80, 28)
    min_area = st.slider("Ignore specks smaller than (px)", 0, 200, 30)
    margin = st.slider("Safety margin (px)", 0, 6, 2)

st.sidebar.subheader("4. Wavelet restoration")
wavelet = st.sidebar.selectbox("Wavelet", ["db4", "sym4", "coif2", "db2", "bior2.2", "haar"])
level = st.sidebar.slider("Decomposition levels", 1, 4, 3)
rule = st.sidebar.radio("Threshold rule", ["bayes", "visu"], horizontal=True,
                        format_func=lambda r: "BayesShrink" if r == "bayes" else "VisuShrink")
tmode = st.sidebar.radio("Thresholding", ["soft", "hard"], horizontal=True)
gain = st.sidebar.slider("Detail gain", 1.0, 1.3, 1.0, 0.05, help="Values above 1 boost surviving detail (usually lowers PSNR).")

st.sidebar.subheader("5. Reconstruction")
method = st.sidebar.radio("Inpainting", ["telea", "ns"], horizontal=True,
                          format_func=lambda m: "Telea" if m == "telea" else "Navier–Stokes")
radius = st.sidebar.slider("Inpainting radius (px)", 1, 15, 5)
show_enh = st.sidebar.checkbox("Also show display enhancement (CLAHE + sharpen)")

# ----------------------------------------------------------------- header
st.title("Wavelet Transform Based Restoration of Damaged Chinese Painting")
st.caption("DIP case study · RBU, EN 2026–27 · damage detection → DWT + BayesShrink denoising → inpainting → MSE / PSNR / SSIM")

if data is None:
    st.info("Upload an image in the sidebar to begin." if not simulate or source == "Upload my own"
            else "Reference painting not found in data/.")
    st.stop()

original = load_image(data, max_side)
if original is None:
    st.error("Could not read that file as an image.")
    st.stop()

params = dict(v_thr=v_thr, s_thr=s_thr, tophat_thr=tophat_thr, min_area=min_area, margin=margin,
              wavelet=wavelet, level=level, mode=tmode, rule=rule, gain=gain, radius=radius, method=method)
with st.spinner("Restoring..."):
    if simulate:
        damaged, gt = run_simulation(original, seed, coverage, tears, scratches, noise, fading)
    else:
        damaged, gt = original, None
    res = run_restore(damaged, params)
restored = res["restored"]
mask = res["mask"]

tabs = st.tabs(["Result", "Damage detection", "Wavelet sub-bands", "Method comparison", "About"])

# ----------------------------------------------------------------- Result
with tabs[0]:
    if simulate:
        c1, c2, c3 = st.columns(3)
        c1.image(original, caption="1. Original painting", clamp=True, width="stretch")
        c2.image(damaged, caption="2. Damaged painting (input)", clamp=True, width="stretch")
        c3.image(restored, caption="4. Digitally restored painting", clamp=True, width="stretch")
        d_mse, d_psnr, d_ssim = R.evaluate(original, damaged)
        r_mse, r_psnr, r_ssim = R.evaluate(original, restored)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSNR (dB)", f"{r_psnr:.2f}", f"{r_psnr - d_psnr:+.2f} vs damaged")
        m2.metric("SSIM", f"{r_ssim:.3f}", f"{r_ssim - d_ssim:+.3f} vs damaged")
        m3.metric("MSE", f"{r_mse:.1f}", f"{100 * (r_mse / d_mse - 1):+.1f}%", delta_color="inverse")
        m4.metric("Damaged area", f"{gt.mean() * 100:.2f}%", f"{res['n_regions']} regions detected", delta_color="off")
    else:
        c1, c2, c3 = st.columns(3)
        c1.image(damaged, caption="Your damaged image", clamp=True, width="stretch")
        c2.image(overlay(damaged, mask), caption=f"Detected damage (red, {res['n_regions']} regions)", clamp=True, width="stretch")
        c3.image(restored, caption="Restored", clamp=True, width="stretch")
        before = sk_sigma(damaged, channel_axis=-1, average_sigmas=True)
        after = sk_sigma(restored, channel_axis=-1, average_sigmas=True)
        m1, m2 = st.columns(2)
        m1.metric("Damaged area detected", f"{mask.mean() * 100:.2f}%")
        m2.metric("Estimated noise σ", f"{after:.4f}", f"{after - before:+.4f}", delta_color="inverse")
        st.caption("No clean reference exists for a real photo, so PSNR and SSIM cannot be computed. "
                   "If losses are missed or over-detected, adjust the detection thresholds in the sidebar.")
    if show_enh:
        st.image(R.enhance(restored), caption="Restored + display enhancement (CLAHE + sharpening)", clamp=True, width="stretch")
    st.download_button("Download restored image (PNG)", png_bytes(restored), "restored.png", "image/png")

# ----------------------------------------------------------------- Detection
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    c1.image(res["raw_mask"].astype(np.float32), caption="3.2 Raw damage map", clamp=True, width="stretch")
    c2.image(mask.astype(np.float32), caption="3.3 Final mask (white = damaged)", clamp=True, width="stretch")
    c3.image(overlay(damaged, mask), caption="Segmentation overlay", clamp=True, width="stretch")
    if simulate:
        sc = R.detection_scores(mask, gt)
        cols = st.columns(5)
        for col, (k, v) in zip(cols, sc.items()):
            col.metric(k, f"{v:.3f}")
        cols[4].metric("Detected / true area", f"{mask.mean() * 100:.1f}% / {gt.mean() * 100:.1f}%")
        err = np.zeros(mask.shape + (3,), np.float32)
        err[mask & gt] = [1, 1, 1]; err[mask & ~gt] = [0.92, 0.41, 0.20]; err[~mask & gt] = [0.16, 0.47, 0.84]
        st.image(err, caption="Against ground truth: white = correct, orange = false positive, blue = missed", clamp=True, width=480)

# ----------------------------------------------------------------- Sub-bands
with tabs[2]:
    bands = R.subbands(R.to_f(res["gray"]), wavelet)
    energy = {k: float(np.sum(v ** 2)) for k, v in bands.items()}
    tot = sum(energy.values())
    c = st.columns(4)
    for col, k, label in zip(c, ["LL", "LH", "HL", "HH"], ["approximation", "horizontal detail", "vertical detail", "diagonal detail"]):
        col.image(R.visual_band(bands[k], detail=k != "LL"), caption=f"{k} ({label}) · {100 * energy[k] / tot:.2f}% energy",
                  clamp=True, width="stretch")
    st.markdown(f"**Noise estimate (MAD) per channel:** " + ", ".join(f"{s:.4f}" for s in res["sigmas"])
                + (f"  ·  true σ = {noise:.3f}" if simulate else ""))
    st.markdown("**Thresholds applied (green channel)**")
    st.dataframe(pd.DataFrame(res["thresholds"]).round(4), hide_index=True, width="stretch")

# ----------------------------------------------------------------- Comparison
with tabs[3]:
    if simulate:
        df = run_baselines(original, damaged, gt, mask, res["denoised"], restored, radius)
        st.dataframe(df, hide_index=True, width="stretch")
        st.bar_chart(df.set_index("Method")["PSNR (dB)"], horizontal=True, color="#C0392B")
        st.caption("All methods use the same detected mask. MSE on the 0–255 scale.")
    else:
        st.info("Method comparison needs a clean reference. Switch to “Simulate damage” mode.")

# ----------------------------------------------------------------- About
with tabs[4]:
    st.markdown("""
**Pipeline**
1. **3.1 Pre-processing** – grayscale, Gaussian and median smoothing (used for detection only)
2. **3.2 Damage detection** – bright, unsaturated pixels in HSV (losses) + white top-hat (scratches)
3. **3.3 Segmentation** – morphology, connected components and a safety margin → damage mask
4. **3.4 DWT** – multi-level 2-D discrete wavelet transform of each colour channel
5. **3.5 Thresholding** – MAD noise estimate, BayesShrink threshold per sub-band, soft/hard thresholding
6. **3.6 Inverse DWT** – denoised image from the modified coefficients
7. **3.7 Reconstruction** – inpainting of the masked regions on the denoised image
8. **3.8 Post-processing** – seam smoothing around the filled regions

With the default settings and the reference painting, the app reproduces the case-study results
(PSNR 24.77 dB, SSIM 0.835, MSE 217.0). Small differences can come from library versions.
""")
