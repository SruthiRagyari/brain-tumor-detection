import gradio as gr
import numpy as np
import cv2
import onnxruntime as ort
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
# Load ONNX Model
# ─────────────────────────────────────────────
session = ort.InferenceSession("model/brain_tumor_classifier.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"ONNX model loaded! Input: {input_name}, Output: {output_name}")

# ─────────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────────
def predict(image):
    if image is None:
        return None, "Please upload an image."

    # Preprocess
    img = Image.fromarray(image).resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Run ONNX inference
    predictions = session.run([output_name], {input_name: img_batch})[0]
    pred_idx = np.argmax(predictions[0])
    pred_class = CLASS_NAMES[pred_idx]

    # Confidence dict for Gradio label
    confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(4)}

    # Info text
    info = f"### {pred_class}\n{CLASS_INFO[pred_class]}"

    return confidences, info

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
            output_info = gr.Markdown(label="Tumor Information")

    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_label, output_info]
    )

    gr.HTML("""
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. Not intended for clinical diagnosis.<br>
        Built with EfficientNetB0 · ONNX Runtime · Gradio
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
