import cv2
from ultralytics import YOLO  # YOLO model

# Config 
class Config:
    DEGUG =True 
    weight = 'WEIGHT/model_last(7).pt'                    # last weight
    model = YOLO(weight)                            # YOLO model
    video_path = "Video/new5.mp4"
    cap = cv2.VideoCapture(video_path)        #demo video
    

    # Frame Text
     # Define the font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Reduced font size
    text_color = (0, 0, 0)  # Black text color
    background_color = (255, 255, 255)  # White background color
    font_thickness = 1
