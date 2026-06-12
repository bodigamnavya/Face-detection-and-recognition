import cv2
import os

name = input("enter person name:")
folder = "known_faces"
if not os.path.exists(folder):
    os.makedirs(folder)

    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(f"{folder}/{name}.jpg", frame)
            
            print("Face Saved Successfully")
            break
    camera.release()
    cv2.destroyAllWindows()