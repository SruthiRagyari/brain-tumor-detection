import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-title {
        text-align: center;
        padding: 1rem 0 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
    }

    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }

    .pred-label {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }

    .pred-confidence {
        font-size: 1.3rem;
        color: #a8b2d1;
    }

    .class-bar {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .info-box {
        background: rgba(102, 126, 234, 0.08);
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #ccd6f6;
    }

    .footer {
        text-align: center;
        color: #495670;
        padding: 2rem 0;
        font-size: 0.85rem;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/brain_tumor_classifier.h5")
    return model

@st.cache_resource
def build_gradcam_model(_model):
    """Build Grad-CAM model from the loaded classifier."""
    # Find EfficientNetB0 base inside the model
    base = None
    for layer in _model.layers:
        if isinstance(layer, tf.keras.Model):
            base = layer
            break

    # Find last Conv2D layer
    last_conv = None
    for layer in reversed(base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break

    # Build grad-cam model
    inp = _model.input
    conv_output = base.get_layer(last_conv.name).output
    base_grad = tf.keras.Model(inputs=base.input, outputs=[conv_output, base.output])
    conv_out, base_out = base_grad(inp)

    head_x = base_out
    for hlayer in _model.layers:
        if hlayer == _model.layers[0] or isinstance(hlayer, tf.keras.Model):
            continue
        head_x = hlayer(head_x)

    grad_model = tf.keras.Model(inputs=inp, outputs=[conv_out, head_x])
    return grad_model

# ─────────────────────────────────────────────
# Grad-CAM Functions
# ─────────────────────────────────────────────
def generate_gradcam(img_array, grad_model):
    """Generate Grad-CAM heatmap."""
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.cast(img_array, tf.float32))
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index.numpy(), predictions.numpy()

def create_overlay(original_img, heatmap, alpha=0.4):
    """Create Grad-CAM overlay on the original image."""
    img = np.array(original_img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return img, heatmap_colored, superimposed

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
CLASS_EMOJIS = ['🔴', '🔵', '🟢', '🟡']
CLASS_INFO = {
    'Glioma': 'Gliomas arise from glial cells in the brain. They are the most common primary brain tumors and can range from low-grade (slow-growing) to high-grade (aggressive).',
    'Meningioma': 'Meningiomas develop from the meninges, the membranes surrounding the brain and spinal cord. Most are benign and slow-growing.',
    'No Tumor': 'No tumor detected in this MRI scan. The brain tissue appears normal.',
    'Pituitary': 'Pituitary tumors develop in the pituitary gland at the base of the brain. Most are benign adenomas that can affect hormone production.'
}

# ─────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-title">🧠 Brain Tumor Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a brain MRI scan to detect and classify tumors using deep learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ About")
    st.markdown("""
    This app uses **EfficientNetB0** with transfer learning to classify brain MRI scans into:

    - 🔴 **Glioma**
    - 🔵 **Meningioma**
    - 🟢 **No Tumor**
    - 🟡 **Pituitary**
    """)

    st.divider()
    st.markdown("### 🔬 Model Details")
    st.markdown("""
    | Spec | Value |
    |------|-------|
    | Backbone | EfficientNetB0 |
    | Input | 224×224 RGB |
    | Classes | 4 |
    | Training | 2-phase |
    """)

    st.divider()
    show_gradcam = st.checkbox("Show Grad-CAM", value=True, help="Visualize which regions the model focuses on")
    gradcam_alpha = st.slider("Overlay opacity", 0.1, 0.8, 0.4, 0.05) if show_gradcam else 0.4

    st.divider()
    st.markdown("""
    <div style="color:#495670; font-size:0.8rem;">
    ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. Not intended for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# Main area — File upload
uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="Supported formats: JPG, PNG, BMP, TIFF"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Load model & predict
    with st.spinner("🔄 Loading model..."):
        model = load_model()
        grad_model = build_gradcam_model(model)

    with st.spinner("🧠 Analyzing MRI scan..."):
        predictions = model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(predictions[0])
        pred_class = CLASS_NAMES[pred_idx]
        confidence = predictions[0][pred_idx] * 100

        if show_gradcam:
            heatmap, _, _ = generate_gradcam(img_batch, grad_model)
            original, heatmap_colored, overlay = create_overlay(image, heatmap, gradcam_alpha)

    # ─── Results ───
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📷 Uploaded MRI Scan")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Prediction")
        emoji = CLASS_EMOJIS[pred_idx]
        color = CLASS_COLORS[pred_idx]

        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-label" style="color: {color};">{emoji} {pred_class}</div>
            <div class="pred-confidence">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Probability bars for all classes
        st.markdown("**Class Probabilities:**")
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
            pct = prob * 100
            bar_color = CLASS_COLORS[i]
            is_pred = " ◀" if i == pred_idx else ""
            st.progress(float(prob), text=f"{CLASS_EMOJIS[i]} {cls}: {pct:.1f}%{is_pred}")

    # ─── Info Box ───
    st.markdown(f"""
    <div class="info-box">
        <strong>ℹ️ {pred_class}:</strong> {CLASS_INFO[pred_class]}
    </div>
    """, unsafe_allow_html=True)

    # ─── Grad-CAM Section ───
    if show_gradcam:
        st.markdown("---")
        st.markdown("### 🔥 Grad-CAM Visualization")
        st.caption("Highlights the regions the model focused on for its prediction")

        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            st.image(original, caption="Original (224×224)", use_container_width=True)
        with gcol2:
            st.image(heatmap_colored, caption="Grad-CAM Heatmap", use_container_width=True)
        with gcol3:
            st.image(overlay, caption=f"Overlay → {pred_class}", use_container_width=True)

else:
    # Placeholder when no image uploaded
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; border:2px dashed rgba(102,126,234,0.3); border-radius:16px; background:rgba(102,126,234,0.03);">
            <div style="font-size:4rem; margin-bottom:1rem;">🧠</div>
            <div style="font-size:1.3rem; color:#8892b0; margin-bottom:0.5rem;">Upload a brain MRI image to get started</div>
            <div style="font-size:0.9rem; color:#495670;">Supports JPG, PNG, BMP, TIFF formats</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Built with EfficientNetB0 • TensorFlow • Streamlit</div>', unsafe_allow_html=True)
