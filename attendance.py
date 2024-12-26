import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
from datetime import datetime

# Directory for known faces
known_face_encodings = []
known_face_names = []

# Load known faces from the "images" folder
def load_known_faces():
    images_dir = 'C:\Users\Dell\Downloads\Smart attendance dataset\train'  # directory containing images of people for face recognition
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(images_dir, filename))
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Encode the face from the image
            face_encoding = face_recognition.face_encodings(rgb_image)[0]
            name = filename.split('.')[0]  # name is the filename (without extension)

            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

# Mark attendance in a CSV file
def mark_attendance(name):
    df = pd.read_csv('attendance.csv')
    if name not in df['Name'].values:
        df = df.append({'Name': name}, ignore_index=True)
        df.to_csv('attendance.csv', index=False)
        print(f"{name}")
    else:
        print(f"{name}")

# Initialize the camera and load known faces
def run_attendance_system():
    load_known_faces()
    
    # Start webcam capture
    video_capture = cv2.VideoCapture(0)
    
    # Create a blank CSV if it doesn't exist
    if not os.path.isfile('attendance.csv'):
        df = pd.DataFrame(columns=['Name'])
        df.to_csv('attendance.csv', index=False)
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        # Convert the image from BGR to RGB
        rgb_frame = frame[:, :, ::-1]
        
        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Loop through all detected faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the detected face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            # If there's a match, find the name of the person
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
            
            # Mark attendance for the person
            mark_attendance(name)
        
        # Display the video feed
        cv2.imshow('Video', frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the system
if __name__ == "__main__":
    run_attendance_system()
