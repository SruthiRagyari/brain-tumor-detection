import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_INFO = {
    'Glioma': 'Gliomas arise from glial cells in the brain. They are the most common primary brain tumors and can range from low-grade (slow-growing) to high-grade (aggressive).',
    'Meningioma': 'Meningiomas develop from the meninges, the membranes surrounding the brain and spinal cord. Most are benign and slow-growing.',
    'No Tumor': 'No tumor detected in this MRI scan. The brain tissue appears normal.',
    'Pituitary': 'Pituitary tumors develop in the pituitary gland at the base of the brain. Most are benign adenomas that can affect hormone production.'
}

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
model = tf.keras.models.load_model("model/brain_tumor_classifier.h5")

# Build Grad-CAM model
_base = None
for _layer in model.layers:
    if isinstance(_layer, tf.keras.Model):
        _base = _layer
        break

_last_conv = None
for _layer in reversed(_base.layers):
    if isinstance(_layer, tf.keras.layers.Conv2D):
        _last_conv = _layer
        break

_inp = model.input
_conv_output = _base.get_layer(_last_conv.name).output
_base_grad = tf.keras.Model(inputs=_base.input, outputs=[_conv_output, _base.output])
_conv_out, _base_out = _base_grad(_inp)

_head_x = _base_out
for _hlayer in model.layers:
    if _hlayer == model.layers[0] or isinstance(_hlayer, tf.keras.Model):
        continue
    _head_x = _hlayer(_head_x)

grad_cam_model = tf.keras.Model(inputs=_inp, outputs=[_conv_out, _head_x])
print("Model and Grad-CAM loaded successfully!")

# ─────────────────────────────────────────────
# Grad-CAM Functions
# ─────────────────────────────────────────────
def generate_gradcam(img_array):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_cam_model(tf.cast(img_array, tf.float32))
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

def create_overlay(img_array_224, heatmap, alpha=0.4):
    img = np.uint8(img_array_224 * 255)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed

# ─────────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────────
def predict(image):
    if image is None:
        return None, None, "Please upload an image."

    # Preprocess
    img = Image.fromarray(image).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_batch, verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_class = CLASS_NAMES[pred_idx]

    # Confidence dict for Gradio label
    confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(4)}

    # Grad-CAM
    heatmap, _, _ = generate_gradcam(img_batch)
    overlay = create_overlay(img_array, heatmap)

    # Info text
    info = f"### {pred_class}\n{CLASS_INFO[pred_class]}"

    return confidences, overlay, info

# ─────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
    ),
    title="Brain Tumor Classifier",
    css="""
    .main-header { text-align: center; margin-bottom: 0.5rem; }
    .main-header h1 { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
    }
    .disclaimer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 1rem;
        padding: 0.8rem;
        border: 1px solid #333;
        border-radius: 8px;
        background: rgba(255,255,255,0.03);
    }
    """
) as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>🧠 Brain Tumor Classifier</h1>
        <p style="color: #888; font-size: 1.1rem;">
            Upload a brain MRI scan to detect and classify tumors using deep learning
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Brain MRI",
                type="numpy",
                height=350
            )
            submit_btn = gr.Button("🔍 Analyze MRI", variant="primary", size="lg")

            gr.Markdown("""
            **Supported classes:**
            - 🔴 Glioma
            - 🔵 Meningioma  
            - 🟢 No Tumor
            - 🟡 Pituitary
            """)

        with gr.Column(scale=1):
            output_label = gr.Label(label="Classification Results", num_top_classes=4)
            output_gradcam = gr.Image(label="Grad-CAM Visualization", height=300)
            output_info = gr.Markdown(label="Tumor Information")

    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_label, output_gradcam, output_info]
    )

    gr.Examples(
        examples=[],
        inputs=input_image,
        label="Try an example (upload your own MRI scan above)"
    )

    gr.HTML("""
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. Not intended for clinical diagnosis.<br>
        Built with EfficientNetB0 · TensorFlow · Gradio
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
