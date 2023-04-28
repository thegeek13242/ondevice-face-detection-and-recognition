import os
import gradio as gr
from PIL import Image

def display_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            images.append(Image.open(os.path.join(folder_path, filename)))
    return images

def image_display(folder_path):
    return display_images(folder_path)

iface = gr.Interface(fn=image_display, inputs="text", outputs="image", title="Display Images")
iface.launch()