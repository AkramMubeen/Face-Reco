import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
from utils import load_and_encode_face,mark_attendance

video_capture = cv2.VideoCapture(0)
 
# Define a list of people with their corresponding image paths
people = [
    ("Aamir", "photos/aamir.jpg"),
    ("Akram", "photos/akram.jpeg"),
    ("Ali", "photos/ali.jpeg"),
    ("Imran Khan", "photos/imran.jpg"),
    ("Jaffer", "photos/jaffer.jpeg"),
    ("Javed", "photos/javed.jpg"),
    ("Madni", "photos/madni.jpeg"),
    ("Mana", "photos/mana.jpeg"),
    ("Ovais", "photos/ovais.jpeg"),
    ("Salman", "photos/salman.jpg"),
    ("Shahrukh", "photos/shahrukh.jpg"),
    ("Waqar", "photos/waqar.jpg"),
    ("Wasim", "photos/wasim.jpg"),
    ("Zain", "photos/zain.jpeg")
]

# Load and encode faces using the function
known_face_encodings, known_faces_names = zip(*[load_and_encode_face(image_path, name) for name, image_path in people]) 
known_face_encodings, known_faces_names = list(known_face_encodings), list(known_faces_names)


humans = known_faces_names.copy()
 
face_locations = []
face_encodings = []
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),None,fx=0.25,fy=0.25)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame,face_locations)
    for face_encoding,face_location in zip(face_encodings,face_locations):
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            similarity_percentage = (1 - face_distance[best_match_index]) * 100
            name = known_faces_names[best_match_index].upper()

            if similarity_percentage >= 50:
            
                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                similarity_text = f"{similarity_percentage:.2f}%"
                ns = name + ' Present ' + similarity_text
                cv2.putText(frame,ns, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                print(name)
                mark_attendance(name)

            
    cv2.imshow("Attendence System",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
