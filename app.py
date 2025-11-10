import gradio as gr
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO

# Load pre-trained YOLOv8 model (small or nano)
model = YOLO('yolov8n.pt')

def detect_objects(image_or_url):
    """
    Accepts either a PIL Image or a URL string.
    Returns an annotated image (numpy array) with detections drawn.
    """
    try:
        if isinstance(image_or_url, str):
            response = requests.get(image_or_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = image_or_url.convert("RGB")
    except Exception as e:
        return f"Error loading image: {e}"

    results = model(img)
    annotated = results[0].plot()  # returns a numpy array with bounding boxes drawn
    return annotated

with gr.Blocks() as demo:
    gr.Markdown("# Image Recognition Web App")
    gr.Markdown("Upload an image or paste a URL to an image, and this app will detect objects using a YOLOv8 model.")
    with gr.Tab("Upload Image"):
        input_image = gr.Image(type="pil", label="Upload an image")
        output_image = gr.Image(label="Annotated Image")
        detect_btn = gr.Button("Detect Objects")
        detect_btn.click(fn=detect_objects, inputs=input_image, outputs=output_image)
    with gr.Tab("Image URL"):
        input_url = gr.Textbox(label="Image URL")
        output_url_image = gr.Image(label="Annotated Image")
        detect_url_btn = gr.Button("Detect Objects")
        detect_url_btn.click(fn=detect_objects, inputs=input_url, outputs=output_url_image)
    gr.Markdown("Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).")

if __name__ == "__main__":
    demo.launch()
