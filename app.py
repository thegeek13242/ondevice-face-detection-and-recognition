# import detect
# import gradio

# VIDEO_PATH = 'test4.mp4'

# def object_detection():
#     detect.main(detect.parse_opt())

# if __name__ == '__main__':
with open("logfaces.json", "w") as f:
    f.write("[]")
#     object_detection()


import gradio as gr
from Arcface.app import update_face_data
import os

# Import the predefined run function
from detect import run
from natsort import natsorted


# Define the Gradio interface
def video_processing(video, threshold, ref_face_dir):
    file_paths = [file.name for file in ref_face_dir]
    update_face_data(file_paths)
    # Call the run function with the video and threshold
    run(source=video, classes=0, vid_stride=5, threshold_arcface=threshold)
    # Get the path to the latest experiment folder
    exp_folder = natsorted(os.listdir("runs/detect/"))[-1]
    # Get the path to the output video file
    output_file = natsorted(os.listdir("runs/detect/" + exp_folder))[-1]
    video_file = "runs/detect/" + exp_folder + "/" + output_file
    # Return the output video file as a Gradio OutputFile object
    return video_file


# Define the Gradio interface inputs and outputs
inputs = [
    gr.Video(format="mp4"),
    gr.Number(value=0.8, label="Threshold"),
    gr.UploadButton(
        label="Reference Faces", file_types=["images"], file_count="multiple"
    ),
]
outputs = gr.Video()

# Create the Gradio app
app = gr.Interface(
    fn=video_processing,
    inputs=inputs,
    outputs=outputs,
    title="Video Processing App",
    description="Apply a threshold to a video using a predefined function.",
)

# Run the app
app.launch(share=True)
