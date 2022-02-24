# ---------------------------------- make2explore.com-------------------------------------------------------------------------#
# Project           - No Mask, No Entry. Face Mask Detection based Door/Gate Opening System 
# Created By        - info@make2explore.com
# Version - 1.0
# Last Modified     - 24/02/2022 15:00:00 @admin
# Software          - Python, Thonny IDE, Standard Python Libraries, OpenCV, Keras, TensorFlow etc.
# Hardware          - Raspberry Pi 4 model B, Logitech c270 webcam, i2c LCD, EM-18 RFID Reader, Level Converter, SG-90 Servo
# Sensors Used      - EM-18 RFID Reader, Logitech c270 webcam
# Source Repo       - https://github.com/make2explore/Face-Mask-Detection-based-Door-Gate-Opening-System
# ----------------------------------------------------------------------------------------------------------------------------#

import RPi.GPIO as GPIO
from gpiozero import Servo
import time
import serial

from gpiozero.pins.pigpio import PiGPIOFactory

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import os

from signal import signal, SIGTERM, SIGHUP, pause
from rpi_lcd import LCD

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
BuzzerPin = 26
GPIO.setup(BuzzerPin, GPIO.OUT)
GPIO.output(BuzzerPin, GPIO.HIGH)


servoPIN = 18
factory = PiGPIOFactory()

servo = Servo(12, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000, pin_factory=factory)
servo.min()

Relay1 = 5
Relay2 = 6
GPIO.setup(Relay1, GPIO.OUT)
GPIO.output(Relay1, GPIO.HIGH)
GPIO.setup(Relay2, GPIO.OUT)
GPIO.output(Relay2, GPIO.HIGH)
      
data = serial.Serial(
                    port='/dev/ttyAMA0',
                    baudrate = 9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )
                    #timeout=1 # must use when using data.readline()
                    #)
lcd = LCD()
def safe_exit(signum, frame):
    exit(1)

def TwoBeep():
    GPIO.output(BuzzerPin, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(BuzzerPin, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(BuzzerPin, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(BuzzerPin, GPIO.HIGH)
    
def LongBeep():
    GPIO.output(BuzzerPin, GPIO.LOW)
    time.sleep(1)
    GPIO.output(BuzzerPin, GPIO.HIGH)
    time.sleep(1)
    

def relay_on(pin):
    GPIO.output(pin, GPIO.HIGH)

def relay_off(pin):
    GPIO.output(pin, GPIO.LOW)
    
def relay_temp(pin):
    GPIO.output(pin, GPIO.LOW)
    time.sleep(2)
    GPIO.output(pin, GPIO.HIGH)

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


def detectMask():
    # load our serialized face detector model from disk
    lcd.clear()
    lcd.text("Starting Camera", 1)
    
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    #print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    #print("[INFO] starting video stream...")
    lcd.text("Camera - ON", 2)
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        maskD = False
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            if (mask > withoutMask):
                label = "Mask Detected"
                maskD = True
                color = (0, 255, 0)
                TwoBeep();
                lcd.clear()
                lcd.text("Mask Detected", 1)
                lcd.text("Welcome!!", 2)
                servo.max()
                time.sleep(1)
                servo.min()
                time.sleep(1)
                relay_temp(Relay2)
                time.sleep(1)
                break
            else:
                label = "No Mask Detected"
                lcd.clear()
                lcd.text("No Mask Detected", 1)
                lcd.text("No Mask No Entry", 2)
                color = (0, 0, 255)
                LongBeep()

            # determine the class label and color we'll use to draw
            # the bounding box and text
            #label = "Mask" if mask > withoutMask else "No Mask"
            #color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
        if(maskD):
            break

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    

signal(SIGTERM, safe_exit)
signal(SIGHUP, safe_exit)
lcd.text("Welcome To", 1)
lcd.text("make2explore.com", 2)
time.sleep(2)
lcd.clear()
lcd.text("Face Mask", 1)
lcd.text("Detection System", 2)
time.sleep(2)
lcd.clear()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

try:     
   while 1:
         #x=data.readline()#print the whole data at once
         #x=data.read()#print single data at once
         print ("Place the card")
         lcd.text("Place Your RFID", 1)
         lcd.text("Card on Reader", 2)
         time.sleep(1)
         x=data.read(12)
         x = str(x, 'UTF-8')
         lcd.clear()
         lcd.text("Scanning Card...", 1)
         lcd.text("Checking ID >>>", 2)
         time.sleep(1)
         
         if x=="0900963100AE":
             #print ("Card No - ",x)
             #print ("Welcome User 1")
             lcd.clear()
             lcd.text("Authorized User", 1)
             lcd.text("Hello Adhya", 2)
             TwoBeep();
             relay_temp(Relay1)
             detectMask()
             #print (" ")
             
         elif x=="88001964699C":
             #print ("Card No - ",x)
             #print ("Welcome User 1")
             lcd.clear()
             lcd.text("Authorized User", 1)
             lcd.text("Hello Samihan", 2)
             TwoBeep();
             relay_temp(Relay1)
             detectMask()
             #print (" ")
             
         elif x=="880013E5235D":
             #print ("Card No - ",x)
             #print ("Welcome User 1")
             lcd.clear()
             lcd.text("Authorized User", 1)
             lcd.text("Hello Raj", 2)
             TwoBeep();
             relay_temp(Relay1)
             detectMask()
             #print (" ")
             
         elif x=="880013E5225C":
             #print ("Card No - ",x)
             #print ("Welcome User 1")
             lcd.clear()
             lcd.text("Authorized User", 1)
             lcd.text("Hello Vaidehi", 2)
             TwoBeep();
             relay_temp(Relay1)
             detectMask()
             #print (" ")
             
         elif x=="880019646A9F":
             #print ("Card No - ",x)
             #print ("Welcome User 1")
             lcd.clear()
             lcd.text("Authorized User", 1)
             lcd.text("Hello Mahesh", 2)
             TwoBeep();
             relay_temp(Relay1)
             detectMask()
             #print (" ")
         else:
             #print ("Wrong Card.....")
             lcd.clear()
             lcd.text("Invalid RFID", 1)
             lcd.text("Unauthorized", 2)
             LongBeep();
             lcd.clear()
             lcd.text("    Access", 1)
             lcd.text("    Denied", 2)
             #print (" ")        
         
         #print x

except KeyboardInterrupt:
       GPIO.cleanup()
       data.close()


# ---------------------------------- make2explore.com-------------------------------------------------------------------------#