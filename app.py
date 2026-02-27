import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="SustainX", layout="wide")

st.title("🌱 SustainX - Smart Weed Density Detection")
st.subheader("AI-Based Spray Optimization System")


# ── Helper functions ─────────────────────────────────────────
def detect_weed_density(image_rgb):
    """
    Analyse the uploaded image using HSV colour segmentation to
    compute the real weed (green vegetation) density percentage.
    Returns: density (0-1), weed_mask (binary), highlighted image.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Green vegetation range in HSV
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    weed_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel)

    # Density = fraction of green pixels
    green_pixels = cv2.countNonZero(weed_mask)
    total_pixels = weed_mask.shape[0] * weed_mask.shape[1]
    density = green_pixels / total_pixels

    # Highlighted overlay
    highlighted = image_rgb.copy()
    highlighted[weed_mask > 0] = [255, 0, 0]  # mark weeds in red
    blended = cv2.addWeighted(image_rgb, 0.6, highlighted, 0.4, 0)

    return density, weed_mask, blended


def classify_weed_type(density, hsv_image, weed_mask):
    """
    Classify weed characteristics from the detected green regions
    and return a weed-type label with herbicide / pesticide advice.
    """
    # Average hue & saturation inside weed mask
    masked_hsv = hsv_image[weed_mask > 0]
    if len(masked_hsv) == 0:
        return "No Weeds Detected", "No treatment needed", "N/A"

    avg_hue = masked_hsv[:, 0].mean()
    avg_sat = masked_hsv[:, 1].mean()

    # Simple rule-based weed type classifier
    if avg_hue < 40 and avg_sat > 100:
        weed_type = "Broadleaf Weeds (e.g., Amaranthus, Parthenium)"
        herbicide = "2,4-D or Metsulfuron-methyl (post-emergence)"
        pesticide = {
            "name": "Chlorpyrifos 20% EC",
            "target_pest": "Aphids, cutworms, leaf miners commonly found on broadleaf weeds",
            "dosage": "2.5 mL per litre of water (500 mL/acre)",
            "application": "Foliar spray during early morning or late evening",
            "safety_interval": "14 days before harvest",
            "toxicity": "Moderate (use PPE: gloves, mask)",
        }
    elif avg_hue >= 40 and avg_hue < 60:
        weed_type = "Grassy Weeds (e.g., Cynodon, Echinochloa)"
        herbicide = "Quizalofop-ethyl or Fenoxaprop-p-ethyl"
        pesticide = {
            "name": "Imidacloprid 17.8% SL",
            "target_pest": "Stem borers, plant hoppers associated with grassy weed patches",
            "dosage": "0.5 mL per litre of water (100 mL/acre)",
            "application": "Soil drench or foliar spray at tillering stage",
            "safety_interval": "21 days before harvest",
            "toxicity": "Low-Moderate (bee-toxic – avoid during flowering)",
        }
    elif avg_hue >= 60 and density > 0.35:
        weed_type = "Dense Sedge / Mixed Weeds"
        herbicide = "Halosulfuron-methyl or Bispyribac-sodium"
        pesticide = {
            "name": "Fipronil 5% SC",
            "target_pest": "Broad-spectrum: stem borers, gall midges, rice bugs in dense weed zones",
            "dosage": "2 mL per litre of water (400 mL/acre)",
            "application": "Spray on standing crop; can also be used as seed treatment",
            "safety_interval": "30 days before harvest",
            "toxicity": "Moderate-High (avoid near water bodies)",
        }
    else:
        weed_type = "Light Mixed Vegetation"
        herbicide = "Glyphosate (pre-planting burndown only)"
        pesticide = {
            "name": "Neem Oil 1500 ppm (Azadirachtin)",
            "target_pest": "General preventive: whiteflies, thrips, mites in low-weed areas",
            "dosage": "5 mL per litre of water (1 L/acre)",
            "application": "Foliar spray every 10–15 days as preventive measure",
            "safety_interval": "3 days (organic-safe)",
            "toxicity": "Very Low (organic / bio-pesticide)",
        }

    return weed_type, herbicide, pesticide


def traditional_impact_stats(density):
    """
    Return a dict comparing traditional blanket-spray farming
    with AI-optimised precision spraying.
    """
    # Traditional: fixed blanket spray regardless of weed density
    trad_spray_litres = 200       # litres/acre (blanket)
    trad_cost_per_acre = 1500     # ₹
    trad_crop_damage_pct = 12.0   # average crop damage from over-spraying
    trad_soil_degrade_pct = 8.0   # soil health degradation %

    # AI-optimised: proportional to actual density
    opt_spray_litres = round(200 * density, 1)
    opt_cost_per_acre = round(1500 * density, 2)
    opt_crop_damage_pct = round(12.0 * density, 2)
    opt_soil_degrade_pct = round(8.0 * density, 2)

    return {
        "trad_spray_litres": trad_spray_litres,
        "trad_cost": trad_cost_per_acre,
        "trad_crop_damage": trad_crop_damage_pct,
        "trad_soil_degrade": trad_soil_degrade_pct,
        "opt_spray_litres": opt_spray_litres,
        "opt_cost": opt_cost_per_acre,
        "opt_crop_damage": opt_crop_damage_pct,
        "opt_soil_degrade": opt_soil_degrade_pct,
    }


# ── Main UI ──────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Field Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Field Image", width="stretch")

        st.markdown("---")

        # ── 1. Real weed density from image ─────────────────
        density, weed_mask, blended = detect_weed_density(image_rgb)

        col1, col2 = st.columns(2)
        with col1:
            st.image(weed_mask, caption="Weed Detection Mask", width="stretch")
        with col2:
            st.image(blended, caption="Weed Highlighted (Red Overlay)", width="stretch")

        st.markdown("---")

        # ── 2. Spray recommendation ─────────────────────────
        if density < 0.15:
            spray = "Low Spray 🟢"
        elif density < 0.35:
            spray = "Medium Spray 🟡"
        else:
            spray = "High Spray 🔴"

        st.subheader("📊 Weed Density Analysis (Image-Based)")
        st.metric("Weed Density", f"{round(density * 100, 2)} %")
        st.write(f"**Recommended Spray Level:** {spray}")

        st.markdown("---")

        # ── 3. Weed type & herbicide / pesticide advice ─────
        hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        weed_type, herbicide, pesticide = classify_weed_type(density, hsv_image, weed_mask)

        st.subheader("🔬 Weed Identification & Treatment")
        st.write(f"**Detected Weed Type:** {weed_type}")
        st.write(f"**Recommended Herbicide:** {herbicide}")

        # ── Detailed pesticide card ─────────────────────────
        if isinstance(pesticide, dict):
            st.markdown("#### 🧪 Recommended Pesticide")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.success(f"**{pesticide['name']}")
                st.write(f"**Target Pest:** {pesticide['target_pest']}")
                st.write(f"**Dosage:** {pesticide['dosage']}")
            with p_col2:
                st.write(f"**Application:** {pesticide['application']}")
                st.write(f"**Safety Interval:** {pesticide['safety_interval']}")
                st.write(f"**Toxicity Level:** {pesticide['toxicity']}")
        else:
            st.write(f"**Recommended Pesticide:** {pesticide}")

        st.markdown("---")

        # ── Full weed-to-pesticide reference table ──────────
        st.subheader("📋 Pesticide Reference Guide (All Weed Types)")
        ref_data = [
            {
                "Weed Type": "Broadleaf (Amaranthus, Parthenium)",
                "Pesticide": "Chlorpyrifos 20% EC",
                "Target Pest": "Aphids, cutworms, leaf miners",
                "Dosage": "2.5 mL/L (500 mL/acre)",
                "Toxicity": "Moderate",
            },
            {
                "Weed Type": "Grassy (Cynodon, Echinochloa)",
                "Pesticide": "Imidacloprid 17.8% SL",
                "Target Pest": "Stem borers, plant hoppers",
                "Dosage": "0.5 mL/L (100 mL/acre)",
                "Toxicity": "Low-Moderate",
            },
            {
                "Weed Type": "Dense Sedge / Mixed",
                "Pesticide": "Fipronil 5% SC",
                "Target Pest": "Stem borers, gall midges, rice bugs",
                "Dosage": "2 mL/L (400 mL/acre)",
                "Toxicity": "Moderate-High",
            },
            {
                "Weed Type": "Light Mixed Vegetation",
                "Pesticide": "Neem Oil 1500 ppm",
                "Target Pest": "Whiteflies, thrips, mites",
                "Dosage": "5 mL/L (1 L/acre)",
                "Toxicity": "Very Low (Organic)",
            },
        ]
        st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── 4. Traditional vs AI-Optimised impact ───────────
        stats = traditional_impact_stats(density)

        st.subheader("🌾 Traditional vs AI-Optimised Crop Impact")

        t_col, o_col = st.columns(2)
        with t_col:
            st.markdown("#### Traditional (Blanket Spray)")
            st.write(f"Spray Volume: **{stats['trad_spray_litres']} litres/acre**")
            st.write(f"Cost per Acre: **₹{stats['trad_cost']}**")
            st.write(f"Crop Damage: **{stats['trad_crop_damage']} %**")
            st.write(f"Soil Degradation: **{stats['trad_soil_degrade']} %**")
        with o_col:
            st.markdown("#### AI-Optimised (Precision Spray)")
            st.write(f"Spray Volume: **{stats['opt_spray_litres']} litres/acre**")
            st.write(f"Cost per Acre: **₹{stats['opt_cost']}**")
            st.write(f"Crop Damage: **{stats['opt_crop_damage']} %**")
            st.write(f"Soil Degradation: **{stats['opt_soil_degrade']} %**")

        st.markdown("---")

        # ── 5. Cost optimisation summary ────────────────────
        normal_cost = stats["trad_cost"]
        optimized_cost = stats["opt_cost"]
        saving = normal_cost - optimized_cost

        st.subheader("💰 Cost Optimization")
        c1, c2, c3 = st.columns(3)
        c1.metric("Normal Spray Cost", f"₹{normal_cost}")
        c2.metric("Optimized Spray Cost", f"₹{round(optimized_cost, 2)}")
        c3.metric("Estimated Savings", f"₹{round(saving, 2)}")

        st.markdown("---")

        # ── 6. Comparison charts ────────────────────────────
        st.subheader("📈 Visual Comparisons")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Cost
        axes[0].bar(["Traditional", "AI-Optimised"],
                     [normal_cost, optimized_cost],
                     color=["#e74c3c", "#2ecc71"])
        axes[0].set_ylabel("Cost (₹)")
        axes[0].set_title("Spray Cost")

        # Crop damage
        axes[1].bar(["Traditional", "AI-Optimised"],
                     [stats["trad_crop_damage"], stats["opt_crop_damage"]],
                     color=["#e74c3c", "#2ecc71"])
        axes[1].set_ylabel("Damage (%)")
        axes[1].set_title("Crop Damage")

        # Spray volume
        axes[2].bar(["Traditional", "AI-Optimised"],
                     [stats["trad_spray_litres"], stats["opt_spray_litres"]],
                     color=["#e74c3c", "#2ecc71"])
        axes[2].set_ylabel("Litres / Acre")
        axes[2].set_title("Spray Volume")

        fig.tight_layout()
        st.pyplot(fig)

    else:
        st.error("Error loading image.")

else:
    st.info("Please upload a field image to start analysis.")
