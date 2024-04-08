import cv2
from config.config import Config
from utils.putText import text
from utils.confidence import conFidence
import torch



def videoCapture(cap):
   if not cap.isOpened():
      print('Error')
   
   while True:
      ret, frame = cap.read()
        
      if not ret:
         print('Error could not read')
         break 
         

      # Suitable frame format for pytorch
      # frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      
      # frame = cv2.resize(frame, (100,100))
      # Performaning Object Detection
      results = Config.model(source=frame,stream=True,show=True,conf=0.5)

      # Check Confidence of Bounding Box on Confidence.py
      conFidence(results)   
      

      key = cv2.waitKey(1)
      if key == ord('q'):
         break
   
   cap.release()
   cv2.destroyAllWindows()
