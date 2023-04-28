# import detect
# import gradio

# VIDEO_PATH = 'test4.mp4'

# def object_detection():
#     detect.main(detect.parse_opt())

# if __name__ == '__main__':
#     with open("logfaces.json","w") as f:
#         f.write("[]")
#     object_detection()


import gradio as gr
import os

# Import the predefined run function
from detect import run
from natsort import natsorted

# Define the Gradio interface
def video_processing(video, threshold):
    # Call the run function with the video and threshold
    run(source=video, classes=0, vid_stride=5, threshold_arcface=threshold)
    # Get the path to the latest experiment folder
    exp_folder = natsorted(os.listdir('runs/detect/'))[-1]
    # Get the path to the output video file
    output_file = natsorted(os.listdir('runs/detect/' + exp_folder))[-1]
    video_file = "runs/detect/" + exp_folder + "/" + output_file + ".mp4"
    # Return the output video file as a Gradio OutputFile object
    return gr.outputs.Video(video_file)

# Define the Gradio interface inputs and outputs
inputs = [
    gr.inputs.Video(type='mp4'),
    gr.inputs.Number(minimum=0, maximum=255, default=128, label='Threshold'),
]
outputs = gr.outputs.Video()

# Create the Gradio app
app = gr.Interface(
    fn=video_processing,
    inputs=inputs,
    outputs=outputs,
    title='Video Processing App',
    description='Apply a threshold to a video using a predefined function.'
)

# Run the app
app.launch()


