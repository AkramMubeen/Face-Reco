import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

def load_and_encode_face(image_path, name):
    """
    Load an image, detect and encode the face, and return the encoding with the corresponding name.

    Parameters:
    - image_path (str): Path to the image file.
    - name (str): Name associated with the person in the image.

    Returns:
    - face_encoding: Encoded face of the person in the image.
    - name (str): Name associated with the person.
    """
    try:
        # Load the image file
        image = face_recognition.load_image_file(image_path)

        # Face detection
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            raise ValueError("No face found in the image.")

        # Encode the face
        face_encoding = face_recognition.face_encodings(image)[0]

        return face_encoding, name

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None


def mark_attendance(name, csv_prefix='', csv_folder='./'):
    # Get the current date and time
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Generate the CSV file name based on the current date
    csv_filename = f"{csv_folder}{csv_prefix}{current_date}.csv"

    # Check if the CSV file exists, create it if not
    try:
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header is None or 'Name' not in header or 'Time' not in header:
                raise FileNotFoundError
    except FileNotFoundError:
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Time'])

    # Check if the name is already present in the attendance record
    with open(csv_filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            if row and row[0] == name:
                print(f"{name} is already marked present at {row[1]}.")
                return

    # Append the attendance record with the new entry
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, current_time])

    print(f"{name} marked present at {current_time} on {current_date}. Attendance recorded in {csv_filename}.")



