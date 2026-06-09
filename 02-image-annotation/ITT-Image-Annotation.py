import os
import cv2
import numpy as np
from tkinter.filedialog import askdirectory
import json
import hashlib
import uuid
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

WINDOW_NAME = "Preview Window"
TRANFORMED_WINDOW_NAME = "Transformed Image"
MODELPATH = "03-camera-app/hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODELPATH)
options = vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands = 2
)
detector = vision.HandLandmarker.create_from_options(options)

gestures = ["like","dislike","stop","rock","peace"]

class ImageTransformer:
    def __init__(self):
        self.username = input("Enter your name for the userId: ")  
        if self.username:
            self.userId = hashlib.sha256((self.username + "saltyMcSaltThe42").encode()).hexdigest() #Just if someone in the hagrid annotations used name => hash for userId to avoid collisions

        self.input_path =  self.select_Direct()
        self.output_path = os.path.join(self.input_path, "_annotations")
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.trans_img = None
        self.boundingBox = []
        self.otherHand = False
        self.myJson = None

        self.gesturecounter = 0
        self.currentGesture = gestures[self.gesturecounter]
        print(f"Now make a picture of '\033[1m{gestures[self.gesturecounter]}\033[0m' | Press 'c' to skip gesture")

        video_id = 0
        self.cap = cv2.VideoCapture(video_id)#
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.currentCapImage = None       

    def get_Bounding_Box(self, frame, handmarks):
        height, width, _ = frame.shape
        BOUNDING_BOX_PADDING = 40
        xValues = []
        yValues = []
        returns = []

        for hand in handmarks:
            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
               
                xValues.append(x)
                yValues.append(y)

            xMin = max(min(xValues) - BOUNDING_BOX_PADDING - 20,0)
            xMax = min(max(xValues) + BOUNDING_BOX_PADDING + 20,width) 
            yMin = max(min(yValues) - BOUNDING_BOX_PADDING,0)
            yMax = min(max(yValues) + BOUNDING_BOX_PADDING, height)
            croppedFrame = frame[yMin:yMax, xMin:xMax]

            if len(returns) == 0 or returns[0][1][0] > xMin:
                returns.append((croppedFrame,(xMin, xMax, yMin, yMax)))
            else:
                returns.insert(0,(croppedFrame,(xMin, xMax, yMin, yMax)))

            xValues.clear()
            yValues.clear()
        return returns
       
    def saveJsonDirectly(self, entry):
        fileName = f"{self.currentGesture}.json"
        allFileName = f"annot-[{self.username}].json"
        savePath = os.path.join(self.output_path, fileName)
        allSavePath = os.path.join(self.output_path, allFileName)
        allJson = entry.copy()

        if os.path.exists(savePath):
            try:
                with open(savePath, "r", encoding="utf-8") as f:
                    existingJson = json.load(f)
                entry = {**existingJson, **entry}
            except:
                pass
        
        if os.path.exists(allSavePath):
            try:
                with open(allSavePath, "r", encoding="utf-8") as f:
                    existingAllJson = json.load(f)
                allJson = {**existingAllJson, **entry}
            except:
                pass
        
        print(savePath)

        with open(savePath, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=4)
        with open(allSavePath, "w", encoding="utf-8") as f:
           json.dump(allJson, f, indent=4)


    def setImageDirect(self, frame):
        self.currentCapImage = frame
        self.img = frame
        self.img_copy = self.img.copy()
        self.trans_img = None
        self.points = []

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_srgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = detector.detect(image_srgb)

        completeWidth =  self.img_copy.shape[1]
        completeHeight =  self.img_copy.shape[0]
        
        if results.hand_landmarks:
            returns = self.get_Bounding_Box(frame, results.hand_landmarks)
            if len(returns) > 1:
                self.otherHand = True
            for i, (cropedImage, box) in enumerate(returns):
                x1, x2, y1, y2 = box
                width = (x2-x1)
                height = (y2-y1)
                boundingBox = [
                    x1 / completeWidth,
                    y1 / completeHeight,
                    width / completeWidth,
                    height / completeHeight
                ]
                self.boundingBox.append(boundingBox)
                cv2.rectangle(self.img_copy,(x1,y1),(x2,y2),(0, 255, 0),4)
                
                name = TRANFORMED_WINDOW_NAME + f"{i}"

                self.trans_img = self.img[y1:y2, x1:x2]
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(name, int(1980/2) + 50, 100)

                preview_resize_factor = 1
                while width*preview_resize_factor > 1920/2 or height*preview_resize_factor > 1080/2:
                    preview_resize_factor -= 0.05
                cv2.resizeWindow(name, (int(width*preview_resize_factor), int(height*preview_resize_factor)))
                cv2.imshow(name, self.trans_img)
        else:
            print("NO HANDS!")
            self.currentCapImage = None
            return
        
        cv2.imshow(WINDOW_NAME, self.img_copy)

    def select_Direct(self):
        path = askdirectory(title="Select Directory")
        if path:
            return path
        else:
            print("No file selected, exiting")
            os._exit(1)

    def handle_save_Direct(self):    
        imgName = str(uuid.uuid4())   
        leading_hand_input = int(input("Which hand is shown? Type => Left (1) | Right (2): "))
        desc = [self.currentGesture]

        if leading_hand_input not in (1, 2):
                    print("Wrong entry")
                    return False
        
        leading_hand = "left" if leading_hand_input == 1 else "right"
        #I assume it goes left => right, so if left is shown hand, gesture is 0 and no_gesture is 1 (and the other way around)
        #I could add an input to give a specific label if 2 gestures are seen, but that would destroy the flow (one has to press space with one hand anyway, so i assume any other hand is from someone else and there can only be 2)
        if self.otherHand:
            if leading_hand == "left":
                desc.append("no_gesture")
            if leading_hand == "right":
                desc.insert(0, "no_gesture")
        
        entry = {
            imgName : {
                    "bboxes": self.boundingBox,
                    "labels": desc,
                    "landmarks": [],
                    "leading_conf": 1.0,
                    "leading_hand": leading_hand,
                    "user_id": self.userId
                }
        }

        self.saveJsonDirectly(entry)

        savepath = os.path.join(self.input_path,self.currentGesture)
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        filepath = os.path.join(savepath, f"{imgName}.jpg")
        print(filepath)
        cv2.imwrite(filepath, self.img)

        return True

    def nextGesture(self):
        self.currentCapImage = None
        self.boundingBox = []
        self.otherHand = False
        self.gesturecounter +=1
        self.gesturecounter = self.gesturecounter % len(gestures)
        self.currentGesture = gestures[self.gesturecounter]
        print(f"ATTENTION!")
        if self.gesturecounter == 0:
            print("NEW ROUND STARTS, CHANGE YOUR LOCATION PLEASE!\n")
        print(f"Now make a picture of '\033[1m{gestures[self.gesturecounter]}\033[0m' | Press 'c' to skip gesture")

    def destroy_previews(self):
        #destoryWindow throws error if window isnt open
        for i in range(3):
            try:
                cv2.destroyWindow(TRANFORMED_WINDOW_NAME + f"{i}")
            except:
                pass

    def run(self):
        while True:
            if self.currentCapImage is None:
                ret, frame = self.cap.read()
                if ret:
                    cv2.imshow(WINDOW_NAME,frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27: #27 is escape
                self.destroy_previews()
                self.currentCapImage = None
                self.boundingBox = []
                self.trans_img = None

            if key == 32 and self.currentCapImage is None:
                print("Taking picture")
                self.setImageDirect(frame=frame)

            if key == ord("c"):
                print("Skipping\n---------\n")
                self.nextGesture()

            if key == ord("s"):         
                if self.trans_img is not None:
                    self.destroy_previews()
                    if self.handle_save_Direct():
                            self.nextGesture()                        
                else:
                    print("No transformed image to save")
            if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                os._exit(0)

transformer = ImageTransformer()
transformer.run()