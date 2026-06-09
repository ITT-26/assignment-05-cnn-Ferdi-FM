import argparse
import os
import cv2
import numpy as np
from tkinter.filedialog import askdirectory
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import keras
from enum import IntEnum
import time
from collections import deque, Counter
import math

#Notes:
# - decided on deque for gesturedetection so accidental gestures get reduced, for faster detection i would just need to reduce length
# - a model with all gestures was MUCH better at distingushing features for me
# - I testet it in a lot of settings (brightness, background, webcams), i really hope my experience translates to your testing, there's a reason I originally had confidence at 0.99 and just lowered it to 0.95 so it hopefully works in settings i couldn't replicate

WINDOW_NAME = "Gesture Camera App"
TEST_WINOW_NAME = "Preview Window"
IMG_SIZE = 64
#For Me Model without rotation worked better
ALL_CONDITIONS_MODEL_PATH = f"03-camera-app/gesture_recognition_64.keras" #val-acc was 0.9520
ALL_CONDITIONS_MODEL_PATH_WITH_ROTATION = f"03-camera-app/gesture_recognition_64_with_rotation.keras" #val-acc was 0.9500 - this model was trained with RandomRotation 0.1, but at least for me is less reliable
NEEDED_CONFIDENCE = 0.95 #like previously decribed was at 0.99 at first and worked really well

#This is how i testet different models
#TEST_MODEL_PATH = f"01-hyperparameters/models/gesture_recognition_{IMG_SIZE}.keras"
#TEST_TRANSFER_MODEL_PATH = "01-hyperparameters/models/gesture_dataset_sample/transferLearning_gesture_recognition.keras"

base_options = python.BaseOptions(model_asset_path="03-camera-app/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands = 1
)
detector = vision.HandLandmarker.create_from_options(options)

classnames = ['rock', 'peace', 'ok', 'one', 'dislike', 'like', 'stop', 'fist', 'three', 'two_up'] #should be the order of CONDITIONS in training-Notebook
framePath = "03-camera-app/frame_cropped.png"
#https://studyopedia.com/opencv/apply-sepia-tone-filter-with-opencv/
sepia_kernel = np.array([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168],[0.393, 0.769, 0.189]])

class Predictions(IntEnum):
    NONE = -1
    ROCK = 0
    PEACE = 1
    OK = 2
    ONE = 3
    DISLIKE = 4
    LIKE = 5
    STOP = 6
    FIST = 7
    THREE = 8
    TWO_UP = 9

class CameraApp:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_path", "--o", type=str, default=None)
        parser.add_argument("--countDown", "--c", type=int, default=None)
        args = parser.parse_args()

        self.output_path = args.output_path
        if self.output_path is None or not os.path.isdir(self.output_path):
            print("No Path or didn't point to any directory")
            self.output_path =  self.select_Directory()

        self.countDown = args.countDown
        if self.countDown is None:
            self.countDown = 3

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        #Gesture-Booleans
        self.countDownEndTime = None
        self.flowerFrameShown = False
        self.portraitModeActive = False
        self.sepiaFilterShown = False
        self.vignetteActive = False
        self.takePhoto = False

        self.newDetectionTime = time.perf_counter() #pref_counter is GPT suggestion instead of time.time()
        self.drawBoundingBox = False

        self.predictionBuffer = deque(maxlen=10)
        self.predictPercent = 0
        self.rawPrediction = Predictions.NONE
        self.currentGesture = Predictions.NONE

        self.resizeFactor = 1
        video_id = 0
        self.cap = cv2.VideoCapture(video_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        self.portraitCrop = self.getPortraitCrop(actual_width, actual_height)
        self.setWindowSizes(actual_height, actual_width) 

        self.currentCapImage = None
        
        #Calculating overlay once in the bgeinning brings a huge performance boost compared to doing it every frame
        portrait_width = int(actual_height * 9 / 16)
        self.standardOverlay = cv2.imread(framePath, cv2.IMREAD_UNCHANGED)
        self.portaitOverlay = cv2.resize(self.standardOverlay, (int(portrait_width), int(actual_height)))
        self.standardOverlay = cv2.resize(self.standardOverlay, (int(actual_width), int(actual_height)), interpolation=cv2.INTER_AREA)
        self.model = keras.models.load_model(ALL_CONDITIONS_MODEL_PATH)
        self.run()
    
    def setWindowSizes(self, height, width):
        self.resizeFactor = 1
        while height*self.resizeFactor > 580:
            self.resizeFactor -= 0.1

        cv2.resizeWindow(WINDOW_NAME, int(width*self.resizeFactor), int(height*self.resizeFactor))
        cv2.moveWindow(WINDOW_NAME, 20, 20)

    def getPortraitCrop(self, actual_width, actual_height):
        portrait_width = int(actual_height * 9 / 16)
        
        x_start = (actual_width - portrait_width) // 2
        x_end = x_start + portrait_width

        return (int(x_start), int(x_end))

    def predictGesture(self, croppedFrame, width):
        img = cv2.resize(croppedFrame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        if self.drawBoundingBox:
            cv2.namedWindow(TEST_WINOW_NAME, cv2.WINDOW_NORMAL)
            cv2.moveWindow(TEST_WINOW_NAME, int(width*self.resizeFactor)+40, 20)
            cv2.resizeWindow(TEST_WINOW_NAME, 400, 400)
            cv2.imshow(TEST_WINOW_NAME, img)
            
        img = img.astype("float32") / 255.0
        img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

        predictions = self.model.predict(img,verbose=0)
        confidence = np.max(predictions[0])
        predicted_class = np.argmax(predictions[0])
        
        self.predictPercent = int(confidence*100)
        self.rawPrediction = Predictions(predicted_class)
        if confidence < NEEDED_CONFIDENCE: 
            predicted_class = -1

        return  Predictions(predicted_class)

    def setImageDirect(self, frame):
        if frame is None:
            return
        whiteImage = np.full(frame.shape, 255, dtype=np.uint8)
        cv2.imshow(WINDOW_NAME, whiteImage)
        cv2.waitKey(1) 

        self.currentCapImage = frame.copy()
        time.sleep(0.25)
        self.drawInfo(frame, True)
        cv2.imshow(WINDOW_NAME, frame)

    def select_Directory(self):
        path = askdirectory(title="Select Directory")
        if path:
            return path
        else:
            print("No Directory selected, exiting")
            os._exit(1)

    def get_Bounding_Box(self, frame, handmarks):
        height, width, _ = frame.shape
        #imgCopyForMarkers = frame.copy()
        BOUNDING_BOX_PADDING_X = 40 
        BOUNDING_BOX_PADDING_Y = 50
        xValues = []
        yValues = []
        returns = []

        for hand in handmarks:
            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                #cv2.circle(imgCopyForMarkers, (x,y), 5, (0,255,0), -1)
                xValues.append(x)
                yValues.append(y)

            #Tried changing to a square bounding box, but it COMPLETLY ruins prediction!
            xMin = max(min(xValues) - BOUNDING_BOX_PADDING_X, 0)
            xMax = min(max(xValues) + BOUNDING_BOX_PADDING_X, width) 
            yMin = max(min(yValues) - BOUNDING_BOX_PADDING_Y, 0)
            yMax = min(max(yValues) + BOUNDING_BOX_PADDING_Y, height)

            croppedFrame = frame[yMin:yMax, xMin:xMax]
            #croppedMarkers = imgCopyForMarkers[yMin:yMax, xMin:xMax]

            #so left hand is always 0 in array (so i could easily change num_hands from mediapipe to 2)
            if len(returns) == 0 or returns[0][1][0] > xMin:
                returns.append((croppedFrame, (xMin, xMax, yMin, yMax)))
            else:
                returns.insert(0,(croppedFrame,(xMin, xMax, yMin, yMax)))

            xValues.clear()
            yValues.clear()

        return returns

    def overlay_png(self, frame):
        overlay = self.standardOverlay if not self.portraitModeActive else self.portaitOverlay
        height, width = overlay.shape[:2]
        roi = frame[:height, :width]
        overlay_rgb = overlay[:, :, :3]
        mask = overlay[:, :, 3]
        cv2.copyTo(overlay_rgb, mask, roi)
        return frame
    
    #https://stackoverflow.com/questions/22654770/creating-vignette-filter-in-opencv/22843077#22843077, chatGPT and myself
    def vignetteImage(self, frame):
        rows, cols = frame.shape[:2]

        kernel_x = cv2.getGaussianKernel(cols, 400)
        kernel_y = cv2.getGaussianKernel(rows, 400)
        mask = kernel_y * kernel_x.T
        mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
        return cv2.convertScaleAbs(frame * mask[:, :, None])
    
    def saveImage(self):
        try:
            existingFiles = len(os.listdir(self.output_path))+1
            savePath = os.path.join(self.output_path, f"transformed_Image-{existingFiles}.jpg")
            while os.path.exists(savePath):
                existingFiles += 1
                savePath = os.path.join(self.output_path, f"transformed_Image-{existingFiles}.jpg")
            cv2.imwrite(savePath, self.currentCapImage)
            print(f"Image saved to: {savePath}")
            return True
        except:
            return False
    
    def predict(self, frame):    
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_srgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = detector.detect(image_srgb)
        
        now = time.perf_counter()
        if results.hand_landmarks:
            returns = self.get_Bounding_Box(frame, results.hand_landmarks)
            if returns:
                for i, (cropedImage, box) in enumerate(returns):
                    x1, x2, y1, y2 = box
                    
                    
                    if i == 0 and now > self.newDetectionTime:
                        self.newDetectionTime = now + 0.1 #only predict every 0.1sec at max, improved performance (prediction took around ~46ms most times)
                        prediction = self.predictGesture(cropedImage, frame.shape[1])
                        self.currentGesture = prediction
                        self.predictionBuffer.append(prediction)
                    
                    if self.drawBoundingBox and not self.takePhoto and self.countDownEndTime is None:
                        cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)

                if len(self.predictionBuffer) >= 10:
                    self.decideAction(frame)
        else:
            self.currentGesture = Predictions.NONE
            self.predictPercent = 0
            self.rawPrediction = Predictions.NONE
            try:
                cv2.destroyWindow(TEST_WINOW_NAME)
            except:
                pass


    def decideAction(self, frame):
        pred, count = Counter(self.predictionBuffer).most_common(1)[0]
        if count >= 8:
            #Fun idea would be to check if twoup and peace are equaly here to simulate a sniping gesture and then cut the image
            #And it does work, but not reliably enough that i want it in the endversion
            if (self.predictionBuffer.count(Predictions.TWO_UP) >= 3 and self.predictionBuffer.count(Predictions.TWO_UP) >= 3):
                print("SNIPPING")

            self.predictionBuffer.clear()
            if pred == Predictions.ROCK:
                self.flowerFrameShown = not self.flowerFrameShown
            if pred == Predictions.ONE:
                self.sepiaFilterShown = not self.sepiaFilterShown
            if pred == Predictions.PEACE:
                self.portraitModeActive = not self.portraitModeActive
                self.applyPortraitMode(frame)
            if pred == Predictions.THREE:
                self.vignetteActive = not self.vignetteActive
            if pred == Predictions.OK and self.countDownEndTime is None:
                self.countDownEndTime = time.time() + self.countDown

    def applyPortraitMode(self, frame):
        height, width = frame.shape[:2]
        portrait_width = int(height * 9 / 16)

        portrait_width = int(portrait_width*self.resizeFactor)
        width = int(width*self.resizeFactor)
        height = int(height*self.resizeFactor)

        if self.portraitModeActive:
            cv2.resizeWindow(WINDOW_NAME, portrait_width, height)
            cv2.moveWindow(WINDOW_NAME, int((width-portrait_width)//2),20)
        else:
            cv2.resizeWindow(WINDOW_NAME, width, height)
            cv2.moveWindow(WINDOW_NAME, 20,20)

    def drawInfo(self, frame, isCapture=False):
        overlay = frame.copy()
        _, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.9
        thickness = 2
        color = (255, 255, 255)

        bar_height = 160 if self.portraitModeActive else 60
      
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        predictionDetailText = f"({self.rawPrediction.name} {self.predictPercent:.0f}%)"

        if not isCapture:
            cv2.putText(
                frame,
                f"{self.currentGesture.name} {predictionDetailText} | B => BBox", #
                (20, 40),
                font, scale, color, thickness, cv2.LINE_AA
            )

        shortcut_text = (
            "ESC => reset | S => save Image"
            if isCapture else
            "Peace => Portrait | Rock => Frame | One => Sepia | Three => Vignette | Ok => take Photo"
            if not self.portraitModeActive else 
            "Peace => Portrait | Rock => Frame;One => Sepia | Three => Vignette;Ok => take Photo" #Todo: Test if space is enough
        )

        lines = shortcut_text.split(";")
        for i,line in enumerate(lines):
            (text_width, text_height), _ = cv2.getTextSize(line,font,scale,thickness)

            placement = (width - text_width - 20, 40 + i*30)
            if self.portraitModeActive:
                placement = (20, 80 + i*30)

            cv2.putText(frame, line, placement, font, scale, color, thickness, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

    def setCountDownText(self, frame, text, countDownSize):
        #https://gist.github.com/xcsrz/8938a5d4a47976c745407fe2788c813a

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, countDownSize, 2)[0]

        cv2.putText(
            img=frame,
            text=text,
            org=(frame.shape[1] // 2 - (textsize[0]//2), frame.shape[0] // 2 + (textsize[1]//2)),
            fontFace=font,
            fontScale=countDownSize,
            color=(255, 255, 255),
            thickness=22,        
        )
        cv2.putText(
            img=frame,
            text=text,
            org=(frame.shape[1] // 2 - (textsize[0]//2), frame.shape[0] // 2 + (textsize[1]//2)),
            fontFace=font,
            fontScale=countDownSize,
            color=(0, 0, 0),
            thickness=16,        
        )

    def run(self):
        countDownSize = 7
        lastText = ""
        while True:
            if self.currentCapImage is None:
                ret, frame = self.cap.read()
                if ret:
                    if self.countDownEndTime is None and not self.takePhoto:
                        #prediction is on unedited frame also means it detects hands on the whole 1080p image even in portraitmode
                        # and only when not counting down or taking picture (freezes preview and doesn't draw bounding box)
                        self.predict(frame) 

                    #Order is important so sepia gets also applied to frame             
                    if self.portraitModeActive:
                        frame = frame[:, self.portraitCrop[0]:self.portraitCrop[1]]      
                    if self.vignetteActive:
                        frame = self.vignetteImage(frame)       
                    if self.flowerFrameShown:
                        frame = self.overlay_png(frame)
                    if self.sepiaFilterShown:
                        frame = cv2.transform(frame, sepia_kernel)
                    
                    if self.countDownEndTime is not None:
                        countDonw = math.ceil(self.countDownEndTime - time.time())
                        if countDonw < 0: 
                            self.takePhoto = True
                            self.countDownEndTime = None       
                            continue

                        text = f"{countDonw:.0f}" if countDonw >= 1 else "Cheese!"
                        
                        if lastText == "" or lastText != text:
                            lastText = text
                            countDownSize = 6.5
                        else:
                            countDownSize = max(3.5, countDownSize * 0.91)
                        self.setCountDownText(frame, text, countDownSize)

                    #if taking photo dont apply GUI
                    if not self.takePhoto:
                        self.drawInfo(frame)

                    if self.takePhoto:
                        self.setImageDirect(frame)
                        self.takePhoto = False 
                    else:
                        cv2.imshow(WINDOW_NAME,frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and self.currentCapImage is not None:
                if self.saveImage():
                    self.currentCapImage = None
        
            if key == ord("b"):
                self.drawBoundingBox = not self.drawBoundingBox
                if not self.drawBoundingBox:
                    try:
                        cv2.destroyWindow(TEST_WINOW_NAME)
                    except:
                        pass

            if key == 27: #27 is escape
                self.currentCapImage = None

            if key == 32 and self.currentCapImage is None: #32 is SPACE
                #self.takePhoto = True
                self.countDownEndTime = time.time() + self.countDown

            if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                self.cap.release()
                os._exit(0)


cameraApp = CameraApp()