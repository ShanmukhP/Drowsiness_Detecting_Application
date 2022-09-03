from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time
from imutils import face_utils
import dlib
import cv2
import time
import threading
import math
from sklearn import tree
import pandas as pd
import numpy as np
import time
from functools import partial
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from threading import Thread
import time

thread_running = True

class MainScreen(Screen):
    pass


class Manager(ScreenManager):
    pass



Builder.load_string('''
<MainScreen>:
    name: "Monitoring"

    FloatLayout:
        Label:
            text: "Monitoring Drowsiness"
            pos_hint: {"x":0.0, "y":0.8}
            size_hint: 1.0, 0.2

        Image:
            # this is where the video will show
            # the id allows easy access
            id: vid
            size_hint: 1, 0.6
            allow_stretch: True  # allow the video image to be scaled
            keep_ratio: True  # keep the aspect ratio so people don't look squashed
            pos_hint: {'center_x':0.5, 'top':0.8}


''')



def training():
    a = pd.read_csv("blinkFatigue.csv")

    features = np.array(a['BPM']).reshape((len(a['BPM']),-1))
    labels = a['FATIGUE']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    return clf


def Euclidean_Distance(x,y):
        dis = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
        return dis

def eye_aspect_ratio(eye):
        A = Euclidean_Distance(eye[1], eye[5])
        B = Euclidean_Distance(eye[2], eye[4])
        C = Euclidean_Distance(eye[0], eye[3])
        D = (A+B)**2
        ear = D / (2.0 * C)

        return ear

def mouth_aspect_ratio(mouth):
        A = Euclidean_Distance(mouth[0], mouth[6])
        A = A**2
        B = Euclidean_Distance(mouth[3], mouth[9])
        B = B**2

        ear = A/B
        return ear



class Main(App):

    def build(self):

        # start the camera access code on a separate thread
        # if this was done on the main thread, GUI would stop
        # daemon=True means kill this thread when app stops
        threading.Thread(target=self.capture, daemon=True).start()

        sm = ScreenManager()
        self.main_screen = MainScreen()
        sm.add_widget(self.main_screen)
        return sm

    def update(self, dt):
        # display image from cam in opencv window
        ret, frame = self.capture.read()
        cv2.imshow("CV2 Image", frame)
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer. 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1


    def capture(self):
        global thread_running
        chances = 0
        drowsy = 0
        siren = 0
        endfps= 0
        startfps=0
        blink = 0
        blink2 = 0
        yawn = 0
        yawn2 = 0
        lastBlink = 0
        blinkDur = 0
        op = 0
        timer1 = 0
        thresh = 3.5
        frame_check = 5
        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
        (minStart, minEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["inner_mouth"]
        (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eyebrow"]
        (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eyebrow"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]


        # make a window for use by cv2
        # flags allow resizing without regard to aspect ratio
        cv2.namedWindow('Hidden', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

        # resize the window to (0,0) to make it invisible
        cv2.resizeWindow('Hidden', 0, 0)
        cap=cv2.VideoCapture(0)
        flag=0
        flag1 = 0
        count = 0
        start = time.time()
        start2 = time.time()

        clf = training()



        while True:

                siren = 0
                ret, frame= cap.read()
                count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                subjects = detect(gray, 0)
                try:
                        subject = list(subjects)[0]
                        shape = predict(gray, subject)
                        shape = face_utils.shape_to_np(shape)#converting to NumPy Array

                        leftEye = shape[lStart:lEnd]
                        leftEAR = eye_aspect_ratio(leftEye)

                        rightEye = shape[rStart:rEnd]
                        rightEAR = eye_aspect_ratio(rightEye)

                        ear = (leftEAR + rightEAR) / 2.0



                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)

                        mouth = shape[mStart:mEnd]
                        mouthHull = cv2.convexHull(mouth)

                        mouthEAR = mouth_aspect_ratio(mouth)


                        nose = shape[nStart:nEnd]
                        noseHull = cv2.convexHull(nose)


                        re = shape[reStart:reEnd]
                        reHull = cv2.convexHull(re)

                        le = shape[leStart:leEnd]
                        leHull = cv2.convexHull(le)

                        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 0), 1)
                        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 0), 1)
                        cv2.drawContours(frame, [mouthHull], -1, (255, 255, 0), 1)
                        cv2.drawContours(frame, [noseHull], -1, (255, 255, 0), 1)
                        #cv2.drawContours(frame, [jawHull], -1, (255, 255, 0), 1)
                        cv2.drawContours(frame, [reHull], -1, (255, 255, 0), 1)
                        cv2.drawContours(frame, [leHull], -1, (255, 255, 0), 1)
                        if ear < thresh:

                                if flag == 0 and time.time()-lastBlink > 1:
                                        blink += 1
                                        lastBlink = time.time()
                                        print("Blink Detected", blink)
                                        

                                print (flag,end=' ')
                                flag += 1
                                if(flag > 10):
                                        cv2.putText(frame, "  STAY ALERT ", (200, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                                        cv2.putText(frame, " DON'T SLEEP ", (200,450),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                                        siren = 1


                        else:
                                flag = 0

                        if(time.time()- start > 60):
                                print("Blink Per minute :",blink)
                                p = np.array([blink]).reshape((1,-1))
                                op = clf.predict(p)
                                print('Chances of Drowsy :', op[0])
                                start = time.time()
                                blink = 0
                                timer1 = 0

                        if op[0] > 0:
                                cv2.putText(frame, "  STAY ALERT ", (200, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                                cv2.putText(frame, " YOU MAYBE SLEEPY ", (200,450),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                                drowsy = 1
                                siren = 1
                                if(timer1 == 0):
                                        start= time.time()
                                        timer1 = 1 

                                elif(timer1 == 1 and ((time.time()- start) > 10)):
                                        op=0


                        if(mouthEAR < 5):
                                flag1 += 1
                                if flag1 > frame_check:
                                        yawn += 1
                                        print("Yawn detected")
                                        flag1 = 0

                        else:
                                flag1 = 0
                except:
                        pass



                endfps = time.time()
                fps = int(1/(endfps-startfps))
                cv2.putText(frame, "FPS:- "+str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

                if(time.time() - start2 > 30):
                        print('Yawns ', yawn)
                        print('BPM : ', blink2)

                        if(yawn > 1 or (3 <= blink <= 4)):
                                cv2.putText(frame, "Chances of Drowsiness Soon", (1, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                                chances = 1
                                siren = 1

                        yawn = 0
                        blink2 = 0
                        start2 = time.time()
                

                startfps = time.time()
                Clock.schedule_once(partial(self.display_frame, frame),1/120)
                #cv2.imshow('Hidden', frame)
                cv2.waitKey(1)
                #cv2.imshow("Sleepiness Monitor", frame)
                #Clock.schedule_interval(self.update, 1.0/33.0)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                        break
                if siren == 1:
                    sound = SoundLoader.load('siren.wav')
                    sound.play()
        cap.release()
        cv2.destroyAllWindows()
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''

    def display_frame(self, frame, dt):
            # display the current video frame in the kivy Image widget
            # create a Texture the correct size and format for the frame
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            # copy the frame data into the texture
            texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
            # flip the texture (otherwise the video is upside down
            texture.flip_vertical()
            # actually put the texture in the kivy Image widget
            self.main_screen.ids.vid.texture = texture



Main().run()
