from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

model = load_model('models/cnncat2.h5')

class CameraApp(App):
    def build(self):
        self.img = Image()
        layout = BoxLayout()
        layout.add_widget(self.img)
        self.cap = cv2.VideoCapture(0)
        self.score = 0
        self.thicc = 2
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return layout

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)
        
        rpred, lpred = 1, 1  # Default to open eyes

        for (x, y, w, h) in right_eye:
            r_eye = gray[y:y+h, x:x+w]
            r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
            r_eye = np.expand_dims(r_eye.reshape(24, 24, -1), axis=0)
            rpred = np.argmax(model.predict(r_eye), axis=-1)[0]
            break
        
        for (x, y, w, h) in left_eye:
            l_eye = gray[y:y+h, x:x+w]
            l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
            l_eye = np.expand_dims(l_eye.reshape(24, 24, -1), axis=0)
            lpred = np.argmax(model.predict(l_eye), axis=-1)[0]
            break
        
        if rpred == 0 and lpred == 0:
            self.score += 1
        else:
            self.score -= 1
        self.score = max(0, self.score)
        
        if self.score > 10:
            try:
                sound.play()
            except:
                pass
            self.thicc = min(10, self.thicc + 2)
        else:
            self.thicc = max(2, self.thicc - 2)
        
        cv2.putText(frame, f'Score: {self.score}', (100, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.score > 10:
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), self.thicc)
        
        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(width, height), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

    def on_stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    CameraApp().run()
