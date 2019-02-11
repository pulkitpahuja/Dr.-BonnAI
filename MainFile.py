import numpy as np
import dlib
import cv2
from math import hypot
import csv
from pygame import mixer
 
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Pulkit Pahuja\Desktop\ML\hackDTU\shape_predictor_68_face_landmarks.dat")

my_irises=[]
d=[]
flag=True
flag1=True
flag2=True
count=0
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

mixer.init()
mixer.music.load(r"C:\Users\Pulkit Pahuja\Desktop\ML\hackDTU\Music\4.mp3")
mixer.music.play()
 
font = cv2.FONT_HERSHEY_PLAIN
def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

 
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
 
    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
 
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
 
    ratio = hor_line_lenght / ver_line_lenght
    
    return ratio



def get_irises_location(frame_gray):
        eye_cascade=cv2.CascadeClassifier( r'C:\Users\Pulkit Pahuja\Desktop\ML\hackDTU\haarcascades\haarcascade_eye_tree_eyeglasses.xml')
        eyes = eye_cascade.detectMultiScale(frame_gray, 1.3, 5)  # if not empty - eyes detected
        
        irises=[]
        for (ex, ey, ew, eh) in eyes:
            iris_w = int(ex + float(ew / 2))
            iris_h = int(ey + float(eh / 2))
            irises.append([np.float32(iris_w), np.float32(iris_h)])
            my_irises.append([np.float32(iris_w), np.float32(iris_h)])

        
            
    
        #print(irises)
        return irises

            
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    irises=get_irises_location(gray)
    
    


    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
 
        landmarks = predictor(gray, face)
 
        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
 
        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
            count=count+1
 
 
        # Gaze detection
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                    (landmarks.part(43).x, landmarks.part(43).y),
                                    (landmarks.part(44).x, landmarks.part(44).y),
                                    (landmarks.part(45).x, landmarks.part(45).y),
                                    (landmarks.part(46).x, landmarks.part(46).y),
                                    (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        
        
        #cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

       
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        height1, width1, _ = frame.shape
        mask1 = np.zeros((height1, width1), np.uint8)
        cv2.polylines(mask1, [right_eye_region], True, 255, 2)
        cv2.fillPoly(mask1, [right_eye_region], 255)
        right_eye = cv2.bitwise_and(gray, gray, mask=mask1)
 
        
         
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        min_x1 = np.min(right_eye_region[:, 0])
        max_x1 = np.max(right_eye_region[:, 0])
        min_y1 = np.min(right_eye_region[:, 1])
        max_y1 = np.max(right_eye_region[:, 1])

 
        gray_eye_left = left_eye[min_y: max_y, min_x: max_x]
        gray_eye_right=right_eye[min_y1: max_y1, min_x1: max_x1]
        
        _, threshold_eye = cv2.threshold(gray_eye_left, 70, 255, cv2.THRESH_BINARY)
 
        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        
        for w,h in irises:
            cv2.circle(frame, (w, h), 7, (0, 255, 0), 2)
            
        
        #print(my_irises)
        
        if (len(my_irises)%100==0):
            d=[]
            for i in range(len(my_irises)-2):
                d.append(distance(np.asarray(my_irises[i+2]),np.asarray(my_irises[i])))
            a=np.mean(d)
            max1=max(d)
            min1=min(d)
            print(a,max1,min1)
            
            if (a<20.0000):
               #mixer.music.stop()
                mixer.music.load(r"C:\Users\Pulkit Pahuja\Desktop\ML\hackDTU\Music\3.mp3")
                mixer.music.play()
                #flag=False
            elif(a>20.0000 and a<30.0000):
                #mixer.music.stop()
                mixer.music.load(r"C:\Users\Pulkit Pahuja\Desktop\ML\hackDTU\Music\2.mp3")
                mixer.music.play()
                #flag1=False
                
            else:
                #mixer.music.stop()
                mixer.music.load(r"C:\Users\Pulkit Pahuja\Desktop\ML\hackDTU\Music\1.mp3")
                mixer.music.play()
                #flag2=False

        
        
        #left_eye_single = cv2.resize(gray_eye_left, None, fx=5, fy=5)
        #right_eye_single = cv2.resize(gray_eye_right, None, fx=5, fy=5)
        #whole_eye=cv2.resize(gray_whole_eye, None, fx=5, fy=5)
        #cv2.imshow("Left Eye", left_eye_single)
        #cv2.imshow("Right Eye", right_eye_single)
        #cv2.imshow("Threshold", threshold_eye)
        #cv2.imshow("Right eye Only",left_eye)

        #cv2.imshow("Left eye Only", right_eye)
        
        
 
    cv2.imshow("Frame", frame)
    
    
    key = cv2.waitKey(1)
    if (key == 27):
        mixer.music.stop()
        break
 
cap.release()
cv2.destroyAllWindows()


#print(my_irises)
d=[]


for i in range(len(my_irises)-2):
    d.append(distance(np.asarray(my_irises[i+2]),np.asarray(my_irises[i])))

#def write_to_csv(data, filename, fieldnames):
 #   with open(filename, "w") as f:
  #      csv_dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
   #     csv_dict_writer.writeheader()
    #    csv_dict_writer.writerows(data)

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
  
# How many elements each 
# list should have 
n = 40
  
#x = list(divide_chunks(d, n))

#print (x)

maxval=max(d)
minval=min(d)

#def deciderinput():
    




    
