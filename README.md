# Ondevice Face Detection & Recognition

We recommend to run the code on **Linux based systems**. There are some conflicts with the dependencies on Windows.

## Gradio Branch

This branch contains the code for the Gradio interface. The main branch is [here](https://github.com/thegeek13242/ondevice-face-detection-and-recognition/tree/main)

## Steps to run

- Clone the repo using `git clone https://github.com/thegeek13242/ondevice-face-detection-and-recognition.git`
- Checkout to the gradio branch using `git checkout gradio`
- Create a virtual environment using `python3 -m venv .venv`
- Activate the virtual environment using `source .venv/bin/activate`
- Install the dependencies using `pip install -r requirements.txt`
- Run the code using `python3 app.py`
- Open the link in the terminal in your browser

## Gradio Interface

### Upload Video File

![Upload Video File](https://i.imgur.com/Q4doXRo.png)

### Set Threshold Value - Recommended value is 0.7

![Set Threshold Value](https://i.imgur.com/IjldjBU.png)

### Reference Image - Upload all the reference images

![Reference Image](https://i.imgur.com/M24Q47R.png)
