# On-device Face Detection & Recognition

We recommend running the code on *Linux-based systems*. There are some conflicts with the dependencies on Windows.

## Main Branch

This branch contains the code for the CLI interface. The gradio branch is [here](https://github.com/thegeek13242/ondevice-face-detection-and-recognition/tree/gradio)

## Steps to run

- Clone the repo using `git clone https://github.com/thegeek13242/ondevice-face-detection-and-recognition.git`
- Checkout to the main branch using `git checkout main`
- Create a virtual environment using `python3 -m venv .venv`
- Activate the virtual environment using `source .venv/bin/activate`
- Install the dependencies using `pip install -r requirements.txt`
- Put reference images in the `face_data` folder. Make sure the `crops` folder exists and `logfaces.json` has `[]` as its content.
- Set threshold value in `main.py` Threshold is the cosine similarity. The lower the number, the higher the restrictions for detection. It will give more accurate results, but may also increase false negatives. Recommended value is 0.7, but maybe increased or decreased accordingly.
- Run the code using `python3 main.py --source "videofile.mp4"`
- Face crops are stored in the `crops` folder and logs are in `logfaces.json`
- `vid_stride` argument is used to specify the frame subsampling rate when processing video inputs. It determines how many frames will be skipped between each frame that is processed by the model. It can be passed to `main.py` as `python3 main.py --source "videofile.mp4" --vid_stride 3`
- Threshold can also be passed as a command line argument as `--threshold_arcface 0.7`