"""
Seismic Fragility Curve Analysis — Streamlit App
=================================================
Upload one or more CSV files containing 'PGA' and 'Disp' columns (results of
nonlinear time-history analyses at different PGA levels), configure damage-
state displacement thresholds and plot styling per dataset, then fit
lognormal CDF fragility curves via Maximum Likelihood Estimation (MLE) and
view/download the resulting plots and fitted parameters.

Run with:
    streamlit run fragility_app.py
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# Page config & constants
# --------------------------------------------------------------------------
st.set_page_config(page_title="Fragility Curve Analysis", layout="wide")

DAMAGE_STATE_DEFAULTS = {
    "Slight": 0.02,
    "Moderate": 0.05,
    "Collapse": 0.10,
}
COLOR_OPTIONS = ["blue", "red", "green", "purple", "orange", "black", "cyan", "magenta", "yellow"]
LINESTYLE_OPTIONS = ["-", "--", ":", "-."]
MARKER_OPTIONS = ["None", "o", "^", "s", "D", "x", "+", "*", "v", "<", ">", "p", "h"]

# --------------------------------------------------------------------------
# Core fitting functions
# --------------------------------------------------------------------------
def lognormal_cdf(pga, median, dispersion):
    return norm.cdf(np.log(pga), np.log(median), dispersion)


def neg_log_likelihood_lognormal(params, pgas, num_occurrences, num_trials):
    median, dispersion = params
    if median <= 0 or dispersion <= 0:
        return np.inf
    probabilities = lognormal_cdf(pgas, median, dispersion)
    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
    log_likelihood = num_occurrences * np.log(probabilities) + (num_trials - num_occurrences) * np.log(
        1 - probabilities
    )
    return -np.sum(log_likelihood)


def fit_fragility(pga_levels, exceed_counts, total_counts):
    """Fit a lognormal fragility curve via MLE. Returns None if no valid data."""
    valid = total_counts > 0
    pga_f = pga_levels[valid]
    exceed_f = exceed_counts[valid]
    total_f = total_counts[valid]
    if len(pga_f) == 0:
        return None

    initial_guess = [np.median(pga_f), 0.5]
    result = minimize(
        neg_log_likelihood_lognormal,
        initial_guess,
        args=(pga_f, exceed_f, total_f),
        method="Nelder-Mead",
        bounds=((1e-6, None), (1e-6, None)),
    )
    median, dispersion = result.x
    empirical = exceed_f / total_f
    return median, dispersion, pga_f, empirical


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "files_data" not in st.session_state:
    st.session_state.files_data = {}
    # Key: filename -> {'content': str,
    #                    'damage_state_settings': {ds_name: {...style...}},
    #                    'damage_state_thresholds': {ds_name: float}}

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📊 Seismic Fragility Curve Analysis")
st.write(
    "Upload CSV file(s) with `PGA` and `Disp` columns (one row per analysis run) "
    "to fit lognormal fragility curves via Maximum Likelihood Estimation."
)

with st.expander("ℹ️ Expected CSV format"):
    st.code("PGA,Disp\n0.1,0.005\n0.1,0.012\n0.2,0.018\n0.2,0.031\n...", language="text")
    st.caption("`PGA` = peak ground acceleration (g) for each run; `Disp` = peak displacement (m) result.")

# --------------------------------------------------------------------------
# File upload
# --------------------------------------------------------------------------
uploaded_files = st.file_uploader("Upload CSV file(s)", type="csv", accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        content = f.getvalue().decode("utf-8")
        if f.name not in st.session_state.files_data:
            ds_settings, ds_thresholds = {}, {}
            for i, (ds_name, default_thresh) in enumerate(DAMAGE_STATE_DEFAULTS.items()):
                color = COLOR_OPTIONS[i % len(COLOR_OPTIONS)]
                ds_settings[ds_name] = {
                    "color": color,
                    "linestyle": LINESTYLE_OPTIONS[0],
                    "label": f"{f.name} - {ds_name}",
                    "marker": "None",
                    "markevery": 10,
                    "plot_empirical": True,
                    "marker_facecolor": color,
                }
                ds_thresholds[ds_name] = default_thresh
            st.session_state.files_data[f.name] = {
                "content": content,
                "damage_state_settings": ds_settings,
                "damage_state_thresholds": ds_thresholds,
            }
        else:
            st.session_state.files_data[f.name]["content"] = content  # refresh on re-upload

col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("🗑️ Clear all files"):
        st.session_state.files_data = {}
        st.rerun()

# --------------------------------------------------------------------------
# Per-file configuration
# --------------------------------------------------------------------------
if not st.session_state.files_data:
    st.info("Upload one or more CSV files above to configure fragility analysis settings.")
else:
    st.subheader("Configure Datasets")

    for filename, file_data in st.session_state.files_data.items():
        with st.expander(f"⚙️ Settings for: {filename}", expanded=True):
            st.markdown("**Damage state thresholds — displacement (m)**")
            thresh_cols = st.columns(len(DAMAGE_STATE_DEFAULTS))
            for i, ds_name in enumerate(DAMAGE_STATE_DEFAULTS.keys()):
                with thresh_cols[i]:
                    file_data["damage_state_thresholds"][ds_name] = st.number_input(
                        ds_name,
                        min_value=0.0,
                        value=float(file_data["damage_state_thresholds"][ds_name]),
                        step=0.005,
                        format="%.4f",
                        key=f"thresh_{filename}_{ds_name}",
                    )

            st.markdown("**Curve styling**")
            for ds_name in DAMAGE_STATE_DEFAULTS.keys():
                s = file_data["damage_state_settings"][ds_name]
                st.caption(f"{ds_name} curve")
                c1, c2, c3, c4, c5, c6, c7 = st.columns([1.6, 1, 1, 1, 1, 1, 1])
                with c1:
                    s["label"] = st.text_input("Label", value=s["label"], key=f"label_{filename}_{ds_name}")
                with c2:
                    s["color"] = st.selectbox(
                        "Color", COLOR_OPTIONS, index=COLOR_OPTIONS.index(s["color"]), key=f"color_{filename}_{ds_name}"
                    )
                with c3:
                    s["linestyle"] = st.selectbox(
                        "Line", LINESTYLE_OPTIONS, index=LINESTYLE_OPTIONS.index(s["linestyle"]), key=f"ls_{filename}_{ds_name}"
                    )
                with c4:
                    s["marker"] = st.selectbox(
                        "Marker", MARKER_OPTIONS, index=MARKER_OPTIONS.index(s["marker"]), key=f"marker_{filename}_{ds_name}"
                    )
                with c5:
                    s["markevery"] = st.number_input(
                        "Mark every", min_value=1, value=int(s["markevery"] or 10), key=f"me_{filename}_{ds_name}"
                    )
                with c6:
                    s["marker_facecolor"] = st.selectbox(
                        "Marker fill",
                        COLOR_OPTIONS,
                        index=COLOR_OPTIONS.index(s["marker_facecolor"]) if s["marker_facecolor"] in COLOR_OPTIONS else 0,
                        key=f"mfc_{filename}_{ds_name}",
                    )
                with c7:
                    s["plot_empirical"] = st.checkbox(
                        "Empirical", value=s["plot_empirical"], key=f"emp_{filename}_{ds_name}"
                    )

    st.divider()

    # ----------------------------------------------------------------------
    # Run analysis
    # ----------------------------------------------------------------------
    run_clicked = st.button("▶️ Run Fragility Analysis", type="primary")

    if run_clicked:
        combined_fig, ax_combined = plt.subplots(figsize=(9, 6))
        ax_combined.set_xlabel("PGA (g)")
        ax_combined.set_ylabel("Probability of Exceedance ($p_f$)")
        ax_combined.set_title("Combined Fragility Curves and Empirical Data")
        ax_combined.grid(True, which="both", ls="-", alpha=0.6)
        ax_combined.set_xlim(left=0, right=1.0)

        ds_figs = {}  # ds_name -> (fig, ax)
        results_summary = []
        errors = []

        for filename, file_data in st.session_state.files_data.items():
            try:
                df = pd.read_csv(io.StringIO(file_data["content"]))
            except Exception as e:
                errors.append(f"Error loading CSV for {filename}: {e}")
                continue

            if "PGA" not in df.columns or "Disp" not in df.columns:
                errors.append(f"{filename}: CSV must contain 'PGA' and 'Disp' columns.")
                continue

            pga_levels = np.sort(df["PGA"].unique())
            if len(pga_levels) == 0:
                errors.append(f"{filename}: no PGA levels found.")
                continue
            pga_for_curve = np.logspace(np.log10(np.min(pga_levels) * 0.5), np.log10(1.0), 100)

            for ds_name, threshold in file_data["damage_state_thresholds"].items():
                s = file_data["damage_state_settings"][ds_name]

                exceed_counts, total_counts = [], []
                for pga in pga_levels:
                    subset = df[df["PGA"] == pga]
                    exceed_counts.append((subset["Disp"] >= threshold).sum())
                    total_counts.append(len(subset))
                exceed_counts = np.array(exceed_counts)
                total_counts = np.array(total_counts)

                fit_result = fit_fragility(pga_levels, exceed_counts, total_counts)
                if fit_result is None:
                    errors.append(f"No valid data points for '{ds_name}' in {filename}.")
                    continue
                median, dispersion, pga_emp, empirical = fit_result

                formatted_label = f"{s['label']} ($\\theta$={median:.4f}, $\\beta$={dispersion:.4f})"
                curve = lognormal_cdf(pga_for_curve, median, dispersion)

                marker = None if s["marker"] == "None" else s["marker"]
                markevery = s["markevery"] if marker else None

                # Combined plot
                ax_combined.plot(
                    pga_for_curve, curve, label=formatted_label, color=s["color"], linestyle=s["linestyle"],
                    marker=marker, markevery=markevery, markerfacecolor=s["marker_facecolor"],
                )
                if s["plot_empirical"]:
                    ax_combined.plot(
                        pga_emp, empirical, marker="o", linestyle="None", color=s["color"],
                        markerfacecolor=s["marker_facecolor"], label=f"{formatted_label} (Empirical)",
                    )

                # Per-damage-state plot
                if ds_name not in ds_figs:
                    fig_ds, ax_ds = plt.subplots(figsize=(9, 6))
                    ax_ds.set_xlabel("PGA (g)")
                    ax_ds.set_ylabel("Probability of Exceedance")
                    ax_ds.set_title(f"Fragility Curves — {ds_name} Damage State")
                    ax_ds.grid(True, which="both", ls="-", alpha=0.6)
                    ax_ds.set_xlim(left=0, right=1.0)
                    ds_figs[ds_name] = (fig_ds, ax_ds)
                fig_ds, ax_ds = ds_figs[ds_name]
                ax_ds.plot(
                    pga_for_curve, curve, label=formatted_label, color=s["color"], linestyle=s["linestyle"],
                    marker=marker, markevery=markevery, markerfacecolor=s["marker_facecolor"],
                )
                if s["plot_empirical"]:
                    ax_ds.plot(
                        pga_emp, empirical, marker="o", linestyle="None", color=s["color"],
                        markerfacecolor=s["marker_facecolor"], label=f"{formatted_label} (Empirical)",
                    )

                results_summary.append(
                    {
                        "Dataset": filename,
                        "Damage State": ds_name,
                        "Threshold (m)": threshold,
                        "Median θ (g)": round(median, 4),
                        "Dispersion β": round(dispersion, 4),
                    }
                )

        for err in errors:
            st.warning(err)

        if results_summary:
            ax_combined.legend(loc="lower right", fontsize=8)
            ax_combined.set_ylim(0, 1)
            combined_fig.tight_layout()

            st.subheader("Combined Fragility Curves")
            st.pyplot(combined_fig)

            buf = io.BytesIO()
            combined_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            st.download_button(
                "⬇️ Download combined plot (PNG)", data=buf.getvalue(),
                file_name="combined_fragility_curves.png", mime="image/png",
            )

            st.subheader("Fit Parameters Summary")
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df, use_container_width=True)
            st.download_button(
                "⬇️ Download summary (CSV)", data=summary_df.to_csv(index=False),
                file_name="fragility_fit_summary.csv", mime="text/csv",
            )

            st.subheader("Individual Damage State Curves")
            for ds_name, (fig_ds, ax_ds) in ds_figs.items():
                ax_ds.legend(loc="lower right", fontsize=8)
                ax_ds.set_ylim(0, 1)
                fig_ds.tight_layout()
                st.pyplot(fig_ds)

                buf_ds = io.BytesIO()
                fig_ds.savefig(buf_ds, format="png", dpi=200, bbox_inches="tight")
                st.download_button(
                    f"⬇️ Download {ds_name} plot (PNG)", data=buf_ds.getvalue(),
                    file_name=f"fragility_{ds_name.lower()}.png", mime="image/png",
                    key=f"dl_{ds_name}",
                )
        else:
            st.error("No fragility curves could be fit. Check your CSV files and settings above.")
