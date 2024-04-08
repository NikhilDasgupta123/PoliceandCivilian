import cv2
from config.config import Config

def frameText(frame_rgb,num_civilians,num_police):
    text = ''
    font = Config.font
    position = Config.font_position
    font_scale = Config.font_scale
    font_color = Config.font_color
    thickness = Config.thickness

    cv2.putText(frame_rgb,num_civilians,(10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 
                cv2.LINE_AA)
    
    cv2.putText(frame_rgb,num_police,(10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0), 2, cv2.LINE_AA)
