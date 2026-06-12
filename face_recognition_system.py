import cv2
import face_recognition
import os
import numpy as np

# Store known face data
known_encodings = []
known_names = []

path = "known_faces"

# Load known faces
for file in os.listdir(path):

    image_path = os.path.join(path, file)

    image = face_recognition.load_image_file(
        image_path
    )

    encodings = face_recognition.face_encodings(
        image
    )

    # Check face exists in image
    if len(encodings) > 0:

        known_encodings.append(
            encodings[0]
        )

        known_names.append(
            os.path.splitext(file)[0]
        )

print("Faces Loaded Successfully")

# Open webcam
video = cv2.VideoCapture(0)

while True:

    success, frame = video.read()

    if not success:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect faces
    face_locations = face_recognition.face_locations(
        rgb_frame
    )

    # Generate encodings
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    # Process each detected face
    for face_encoding, face_location in zip(
        face_encodings,
        face_locations
    ):

        name = "Unknown"

        # Check if known faces exist
        if len(known_encodings) > 0:

            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding
            )

            face_distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_match_index = np.argmin(
                face_distances
            )

            if matches[best_match_index]:
                name = known_names[
                    best_match_index
                ]

        top, right, bottom, left = face_location

        # Draw rectangle
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Face Recognition System",
        frame
    )

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()