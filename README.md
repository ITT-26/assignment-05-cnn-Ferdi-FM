[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cMaQVOgt)

## Requirements
- Python **3.9 - 3.12** (tensorflow/keras don't work on higher versions)

## Initializing and starting Virtual Enviroment

### For Windows
Open The Root-Directory (Assignment-05-...) in a Terminal and create + activate the virtual enviroment with (**make sure you use a supported version**):
````
py -3.12 -m venv venv
venv\Scripts\activate
````
(venv) should now be displayed before your new CommandLine in the Terminal

Next install the requirements:
````
pip install -r requirements.txt
````

### For Mac
The Steps are the same, but the concrete commands different:
````
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````
<hr style="border: 2px solid #444; margin: 60px 0;">

# Table of Contents

| Task | Name |
|-|-|
| 01  | [Hyperparameters](#01-hyperparameters) |
| 02_a | [Dataset](#02_a-dataset) |
| 02_b | [Image Annotation Tool](#02_b-image-annotation-tool) |
| 03  | [Camera App](#03-camera-app) |

# 01 Hyperparameters

Put the notebook(s) into the same folder as the subset of HaGrid (gesture_dataset_sample), for example:
````
gesture_dataset_sample/           
    ├── _annotations/
        ├── dislike.json
        ├── ...
    ├── like.json         
    ├── ....   
models/
hyperparameters.ipynb
````
The plots inside the notebook require the models to be fitted directly, so it didn't really make sense to save the models, one would need to save histories instead  
if interested, "model_training-for-camera app" was used to fit models for Task 3

# 02_a dataset

Folder contains captured images, annotations and an image of the confusion-matrix.  
The annotations are once in the format from the Hagrid-Dataset and once according to the requirements of the Assignemtn **_annot-[name]_**:
````
_annotations/           
    ├── annot-[name].json\
    ├── like.json         
    ├── ....   
dislike/
like/
...
conf-matrix.png
confMatrix_Task_2_Notebook.ipynb
gesture_recognition_64_FOR_TASK_2.keras
````
You can reproduce my results with the contained Notebook (It will load the model and set my data for prediction)

# 02_b Image-annotation Tool
I wrote a script for (almost) complete automatic image annotation with mediapipe

## Workflow
- Start the script
    ````
    py 02-image-annotation\ITT-Image-Annotation.py
    ````
- It will ask for an Input to enter your name (this is also used to generate a UserId with a little salt)
- In the Terminal it will show what gesture you should make. You can press **_C_** to skip to the next gesture (Once all gestures are done it will prompt you to change location)
- Press **_Space_** to capture an image
- Your hand will be auto-detected and boundingBoxes will be automatically drawn (it will max detect 2 Hands)
- If no Hands are detected just spam **_Space_** a bit, with too bright lighting it can be fickle
- Press **_S_** to save the image and an Input will ask you if its the right or left hand, if there's a second Hand in the image it will be labeled as "no_gesture"
- Json will be automatically updated


# 03 Camera-App
- [Summary of Gestures/Shortcuts](#gestures-and-shortcuts)
- Start the script (with optional command parameters --output_path & --countDown)
    ````
    py 03-camera-app\camera_app.py --output_path "example/examplefolder" --countDown 5
    ````
    ````
    py 03-camera-app\camera_app.py
    ````
    If no parameters are given a DirectoryDialog for the Output-path will open and the countDown is set to 3
- Now the Image from your cam is shown, your hand will be automatically recognized so you can just start making gestures (only one hand is recognized at a time). **I recommend having a clear background for your hand**
- Press **_B_**, if you want to display the tracking of your hand with a bounding box and an extra little window that shows the cropped image which will be used for prediction.
- Your current accepted gesture will be displayed in the top left, in bracets beside it is the confidence of the current hand. Possible gestures and their impact are displayed on the top right corner (if in portrait mode they will be stacked)  
To activate an effect: hold the gesture for about a second, this is to prevent accidental inputs.  
To remove an effect: simply repeat the gesture.

### Gestures and Shortcuts

| Gesture | Impact |
|----|----|
| ONE  | Applies a Sepia-Filter |
| ROCK | Will display a pretty frame around the image |
| PEACE | Will crop the frame to portrait-Mode (Gesture detection still works for your whole 16:9 frame! You don't need to have your hand in the cropped frame) |
| THREE | Adds a vignette around the frame |
| OK  | Will start a countdown with your specified countDown time, at which end an image will be taken (With cool effects) |


|Shortcut| Impact|
|----|----|
| B | Will show a bounding box around your hand and open a little preview-window for the predicted image |
| ESC | Will cancel if you're unhappy with a captured image |
| SPACE | Starts the countdown, same as __*OK*__-Gesture |


# Citations:
Frame: <a href="https://www.citypng.com/photo/30170/purple-vertical-chinese-frame-png-img">Purple Vertical Chinese Frame PNG IMG</a>
        
