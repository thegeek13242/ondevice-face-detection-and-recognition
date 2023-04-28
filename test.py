import gradio as gr
import json

# Define the Gradio app interface
def app(json_file):
    # Read the contents of the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Convert the JSON object to a string and return it
    data_str = json.dumps(data)
    return data_str

# Define the Gradio app input interface
input_file = gr.inputs.File(label="Upload JSON file")

# Define the Gradio app output interface
output_text = gr.outputs.Textbox()

# Define the Gradio app interface
gr.Interface(fn=app, inputs=input_file, outputs=output_text).launch()
