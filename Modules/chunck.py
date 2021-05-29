import cv2
import imutils
import dlib
import time
import numpy as np

class Chunck:
    def __init__(self, video_path, shape_predictor_path, size=(300, 270)):
        self.path = video_path
        self.size = size
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.x1_avg_face, self.x2_avg_face = 0, 0
        self.y1_avg_face, self.y2_avg_face = 0, 0
        self.x1_avg_landmark, self.x2_avg_landmark = 0, 0
        self.y1_avg_landmark, self.y2_avg_landmark = 0, 0
        self.angle = 0
        
        cap = cv2.VideoCapture(video_path)
        self.WIDTH = int(cap.get(3))
        self.HEIGHT = int(cap.get(4))
        self.FRAME_COUNT = cap.get(cv2.CAP_PROP_FRAME_COUNT)
     
    
    def prepare(self):
        """
        Метод для подготовки нахождения координат лица, лендмарков
        и вращения, чтобы губы были параллельно оси Ox
        """
        
        cap = cv2.VideoCapture(self.path)
        detections = 0
        self.x1_avg_face, self.x2_avg_face = 0, 0
        self.y1_avg_face, self.y2_avg_face = 0, 0
        self.x1_avg_landmark, self.x2_avg_landmark = 0, 0
        self.y1_avg_landmark, self.y2_avg_landmark = 0, 0
        self.x_max_landmark, self.y_max_landmark = 0, 0
        self.x_min_landmark, self.y_min_landmark = self.WIDTH, self.HEIGHT
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
                   
            for face in faces:
                detections += 1
                self.x1_avg_face += face.left()
                self.x2_avg_face += face.right()
                self.y1_avg_face += face.top()
                self.y2_avg_face += face.bottom()
                
                landmarks = self.predictor(gray, face)
                mouth_landmarks = landmarks.parts()[48:60]
                
                x_min, x_max, y_min, y_max = self.WIDTH, 0, self.HEIGHT, 0
                for i, landmark in enumerate(mouth_landmarks):
                    x, y = landmark.x, landmark.y
                    if i == 0:
                        self.x1_avg_landmark += x
                        self.y1_avg_landmark += y
                    elif i == 6:
                        self.x2_avg_landmark += x
                        self.y2_avg_landmark += y
                        
                    x_min, x_max = min(x, x_min), max(x, x_max)
                    y_min, y_max = min(y, y_min), max(y, y_max)
                    
                self.x_min_landmark = min(self.x_min_landmark, x_min)
                self.x_max_landmark = max(self.x_max_landmark, x_max)
                self.y_min_landmark = min(self.y_min_landmark, y_min)
                self.y_max_landmark = max(self.y_max_landmark, y_max)
                        
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        
        cap.release()
        
        if detections == 0:
            return False

        self.x1_avg_face = int(self.x1_avg_face / detections)
        self.x2_avg_face = int(self.x2_avg_face / detections)
        self.y1_avg_face = int(self.y1_avg_face / detections)
        self.y2_avg_face = int(self.y2_avg_face / detections)
        
        self.x1_avg_landmark = int(self.x1_avg_landmark / detections)
        self.x2_avg_landmark = int(self.x2_avg_landmark / detections)
        self.y1_avg_landmark = int(self.y1_avg_landmark / detections)
        self.y2_avg_landmark = int(self.y2_avg_landmark / detections)
        
        a = self.x2_avg_landmark - self.x1_avg_landmark
        b = self.y2_avg_landmark - self.y1_avg_landmark
        self.angle = 180 * np.arctan2(b, a) / np.pi
        
        self.face = dlib.rectangle(self.x1_avg_face, self.y1_avg_face,
                                  self.x2_avg_face, self.y2_avg_face)
        
        self.mouth = dlib.rectangle(self.x_min_landmark, self.y_min_landmark, 
                                   self.x_max_landmark, self.y_max_landmark)

        return True
        
        
    def __preprocess(self, time_sleep, out_path=None):
        cap = cv2.VideoCapture(self.path)
        out = None
        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(out_path, fourcc, 10, self.size)

        length = 60 - 48
        x_arr, x_prev_arr = np.zeros((length)), np.zeros((length))
        y_arr, y_prev_arr = np.zeros((length)), np.zeros((length))
        velocity_x, velocity_y = 0, 0
        start_iteration = 2
        
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Iteration count which will be useful for turning on velocity counter

        iteration = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            blank = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
            
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = self.predictor(gray, self.face)

            x1_angle, x2_angle, y1_angle, y2_angle = 0, 0, 0, 0
            x1_mouth, x2_mouth, y1_mouth, y2_mouth = self.WIDTH, 0, self.HEIGHT, 0

            current_velocity_x, current_velocity_y = 0, 0
            for i, landmark in enumerate(landmarks.parts()[48:60]):
                x_prev_arr[i], x_arr[i] = x_arr[i], landmark.x
                y_prev_arr[i], y_arr[i] = y_arr[i], landmark.y

                if iteration >= start_iteration:
                    current_velocity_x += np.abs(x_arr[i] - x_prev_arr[i])
                    current_velocity_y += np.abs(y_arr[i] - y_prev_arr[i])


                if i != 0:
                    x_prev, y_prev = x, y
                    x, y = landmark.x, landmark.y
                    cv2.line(blank, (x_prev, y_prev), (x, y), (255, 255, 255), 1)
                else: 
                    x, y = landmark.x, landmark.y

                if i == 0:
                    x1_angle, y1_angle = x, y
                elif i == 6:
                    x2_angle, y2_angle = x, y

                x1_mouth, x2_mouth = min(x1_mouth, x), max(x2_mouth, x)
                y1_mouth, y2_mouth = min(y1_mouth, y), max(y2_mouth, y)

                cv2.circle(frame, (x, y), 1, (0, 0, 255))

            current_velocity_x /= length
            current_velocity_y /= length
            velocity_x += current_velocity_x
            velocity_y += current_velocity_y
                

            # rotate frame around center on angle
            angle = 180 * np.arctan2(y2_angle - y1_angle, x2_angle - x1_angle) / np.pi
            center = ((x2_mouth + x1_mouth) / 2, (y2_mouth + y1_mouth) / 2)
            frame = imutils.rotate(frame, angle, center=center)
            blank = imutils.rotate(blank, angle, center=center)
               
            # Crop mouth area
            height = self.y_max_landmark - self.y_min_landmark
            width = self.x_max_landmark - self.x_min_landmark


            general_center = self.mouth.center()
            shift = (general_center.x - center[0], general_center.y - center[1])

            top = int(self.y_min_landmark - shift[1] - height / 4)
            bottom = int(self.y_max_landmark - shift[1] + height / 4)
            left = int(self.x_min_landmark - shift[0] - width / 4)
            right = int(self.x_max_landmark - shift[0] + width / 4)
            frame = frame[top : bottom, left : right]
            blank = blank[top : bottom, left : right]

            
            # frame = imutils.resize(frame, width=self.size[0], height=self.size[1])
            frame = cv2.resize(frame, self.size)
            # blank = imutils.resize(blank, width=self.size[0], height=self.size[1])
            blank = cv2.resize(blank, self.size)

            if out_path is None:
                cv2.imshow('chunk', frame)
                cv2.imshow('blank', blank)
            
                time.sleep(time_sleep)
            else:
                out.write(blank)
            
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

            iteration += 1
                
        cap.release()

        if out_path is None:
            cv2.destroyAllWindows()

        velocity_x /= (iteration - start_iteration)
        velocity_y /= (iteration - start_iteration)

        velocity_x *= fps
        velocity_y *= fps

        return (velocity_x, velocity_y)


    def show(self, time_sleep=0):
        return self.__preprocess(time_sleep)
        
        
    def to_file(self, filename):
        return self.__preprocess(0, filename)
