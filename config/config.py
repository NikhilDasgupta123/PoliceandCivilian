import cv2
from ultralytics import YOLO  # YOLO model
# import pafy                   # To use You tube Videos

# Config 
class Config:
    DEGUG =True
    cap = cv2.VideoCapture('Video/41.mp4')         #demo video 
    weight = 'WEIGHT/last(4).pt'                 # last weight
    model = YOLO(weight)                     # YOLO model
    font = cv2.FONT_HERSHEY_SIMPLEX                # Frame Font
    font_position = (50, 50)                       # Frame Font Position
    font_scale = 1                                 # Size
    font_color = (255, 0, 0)                       # BGR color
    thickness = 2                                  # Thickness
    # video_url = 'https://youtu.be/CGOyzUtogUQ?si=86q9AGx8g-aBwS9O'
    # video = pafy.new(video_url)


