import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="SustainX", layout="wide")

st.title("🌱 SustainX - Smart Weed Density Detection")
st.subheader("AI-Based Spray Optimization System")


# ─────────────────────────────────────────────
# KANNADA VOICE FUNCTION
# ─────────────────────────────────────────────
def speak_kannada(text):
    tts = gTTS(text=text, lang='kn')
    file_path = "voice_output.mp3"
    tts.save(file_path)
    return file_path


# ─────────────────────────────────────────────
# WEED DENSITY DETECTION
# ─────────────────────────────────────────────
def detect_weed_density(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    weed_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel)

    green_pixels = cv2.countNonZero(weed_mask)
    total_pixels = weed_mask.shape[0] * weed_mask.shape[1]
    density = green_pixels / total_pixels

    highlighted = image_rgb.copy()
    highlighted[weed_mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(image_rgb, 0.6, highlighted, 0.4, 0)

    return density, weed_mask, blended


# ─────────────────────────────────────────────
# WEED CLASSIFICATION
# ─────────────────────────────────────────────
def classify_weed_type(density, hsv_image, weed_mask):
    masked_hsv = hsv_image[weed_mask > 0]
    if len(masked_hsv) == 0:
        return "No Weeds Detected", "No treatment needed", {
            "name": "N/A",
            "target_pest": "N/A",
            "dosage": "N/A",
            "application": "N/A",
            "safety_interval": "N/A",
            "toxicity": "N/A",
        }

    avg_hue = masked_hsv[:, 0].mean()
    avg_sat = masked_hsv[:, 1].mean()

    if avg_hue < 40 and avg_sat > 100:
        weed_type = "Broadleaf Weeds"
        herbicide = "2,4-D"
        pesticide = {
            "name": "Chlorpyrifos 20% EC",
            "target_pest": "Aphids, cutworms",
            "dosage": "2.5 mL per litre",
            "application": "Foliar spray",
            "safety_interval": "14 days",
            "toxicity": "Moderate",
        }
    elif avg_hue < 60:
        weed_type = "Grassy Weeds"
        herbicide = "Quizalofop-ethyl"
        pesticide = {
            "name": "Imidacloprid 17.8% SL",
            "target_pest": "Stem borers",
            "dosage": "0.5 mL per litre",
            "application": "Foliar spray",
            "safety_interval": "21 days",
            "toxicity": "Low-Moderate",
        }
    else:
        weed_type = "Mixed Weeds"
        herbicide = "Glyphosate"
        pesticide = {
            "name": "Neem Oil 1500 ppm",
            "target_pest": "Whiteflies",
            "dosage": "5 mL per litre",
            "application": "Preventive spray",
            "safety_interval": "3 days",
            "toxicity": "Low",
        }

    return weed_type, herbicide, pesticide


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Field Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

        density, weed_mask, blended = detect_weed_density(image_rgb)

        col1, col2 = st.columns(2)
        col1.image(weed_mask, caption="Weed Mask", use_container_width=True)
        col2.image(blended, caption="Highlighted Weeds", use_container_width=True)

        st.metric("🌿 Weed Density", f"{round(density*100,2)} %")

        hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        weed_type, herbicide, pesticide = classify_weed_type(density, hsv_image, weed_mask)

        st.success(f"🪴 Weed Type: {weed_type}")
        st.info(f"🧪 Herbicide: {herbicide}")

        st.subheader("💊 Recommended Pesticide")
        st.write(f"**{pesticide['name']}**")
        st.write(f"Target Pest: {pesticide['target_pest']}")
        st.write(f"Dosage: {pesticide['dosage']}")
        st.write(f"Application: {pesticide['application']}")
        st.write(f"Safety Interval: {pesticide['safety_interval']}")
        st.write(f"Toxicity: {pesticide['toxicity']}")

        st.markdown("---")

        # ─────────────────────────────────────────────
        # DYNAMIC COST CALCULATION
        # ─────────────────────────────────────────────
        st.subheader("💰 Smart Cost Optimization")

        field_area_acre = 1
        spray_per_acre = 200
        cost_per_litre = 7.5

        traditional_spray = spray_per_acre * field_area_acre
        traditional_cost = traditional_spray * cost_per_litre

        optimized_spray = traditional_spray * density
        optimized_cost = optimized_spray * cost_per_litre

        savings = traditional_cost - optimized_cost

        c1, c2, c3 = st.columns(3)
        c1.metric("Normal Spray Cost", f"₹{round(traditional_cost,2)}")
        c2.metric("Optimized Spray Cost", f"₹{round(optimized_cost,2)}")
        c3.metric("Estimated Savings", f"₹{round(savings,2)}")

        st.markdown("---")

        # ─────────────────────────────────────────────
        # GRAPH SECTION
        # ─────────────────────────────────────────────
        st.subheader("📊 Visual Comparison")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].bar(["Traditional", "AI"], [traditional_cost, optimized_cost])
        axes[0].set_title("Cost Comparison")

        axes[1].bar(["Traditional", "AI"], [traditional_spray, optimized_spray])
        axes[1].set_title("Spray Volume")

        axes[2].bar(["Traditional", "AI"], [100, round(density*100,2)])
        axes[2].set_title("Coverage %")

        st.pyplot(fig)

        st.markdown("---")

        # ─────────────────────────────────────────────
        # KANNADA VOICE OUTPUT
        # ─────────────────────────────────────────────
        st.subheader("🔊 ಕನ್ನಡದಲ್ಲಿ ಕೇಳಿ")

        kannada_text = f"""
        ನಿಮ್ಮ ಹೊಲದಲ್ಲಿ ಕಳೆ ಪ್ರಮಾಣ {round(density*100,2)} ಶೇಕಡಾ ಇದೆ.
        ಕಳೆ ಪ್ರಕಾರ {weed_type}.
        ಶಿಫಾರಸು ಮಾಡಿದ ಔಷಧಿ {herbicide}.
        ಕೀಟನಾಶಕ {pesticide['name']}.
        ಸಾಮಾನ್ಯ ಖರ್ಚು {round(traditional_cost,2)} ರೂಪಾಯಿ.
        AI ಬಳಸಿ ಖರ್ಚು {round(optimized_cost,2)} ರೂಪಾಯಿ.
        ನೀವು {round(savings,2)} ರೂಪಾಯಿ ಉಳಿಸಬಹುದು.
        """

        if st.button("🔊 ಫಲಿತಾಂಶವನ್ನು ಕೇಳಿ"):
            audio_file = speak_kannada(kannada_text)
            st.audio(audio_file)

    else:
        st.error("Error loading image.")

else:
    st.info("Please upload a field image to start analysis.")
