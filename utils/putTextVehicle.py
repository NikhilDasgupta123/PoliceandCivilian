import cv2
from config.config import Config
import numpy as np


new_count = 0

# About Object Information
def frameObjectInfo(results):
    total_people_count = 0
    for result in results:
      boxes = result.boxes  # Boxes object for bbox outputs
      probs = result.probs  # Class probabilities for classification outputs
      cls = boxes.cls.tolist()  # Convert tensor to list
      xyxy = boxes.xyxy
      xywh = boxes.xywh  # box with xywh format, (N, 4)
      conf = boxes.conf
      # print(cls) # Class Index Array
      
      
      global new_arr
      new_arr = []
      for class_index in cls:
            if class_index == 2.0 or class_index==0:
                new_arr.append(class_index)
                
            elif class_index == 3.0 or class_index==0:
                new_arr.append(class_index)
            
      print(new_arr)
      fourWheel = 0
      twoWheel = 0
      for i in new_arr:
            if i == 2.0:
                fourWheel += 1
            elif i == 3.0:   
                twoWheel +=1

      # Putting Value in dict People and Civilian Count 
      text_info={
          'Four Wheeler Count' : fourWheel,
          'Two Wheeler Count' : twoWheel
      }

      print('Four Wheeler Count :',fourWheel)
      print('Two Wheeler Count :',twoWheel)

      global total_four_two_wheel
      total_four_two_wheel = fourWheel + twoWheel
      print('Total: ',total_four_two_wheel)
      
      # Total Count People Coutn
      total_four_two_wheel_dict = {'Total Vehicle' : total_four_two_wheel}

      return text_info,total_four_two_wheel_dict




# Putting Text Count People and Civilian
def puttext(annotated_frame,text_info):
    # Display the text upside down at the bottom of the frame
    text_y = annotated_frame.shape[0] - 150  # Starting position at the bottom of the frame
    
    # display the Text
    for i, (key, value) in enumerate(text_info.items()):
        reversed_text = key + ': ' + str(value)
        text_size = cv2.getTextSize(reversed_text, Config.font, Config.font_scale, Config.font_thickness)[0]
        # text_x = (annotated_frame.shape[1] - text_size[0]) // 2  # Center the text horizontally
        text_x = 10
        # Draw white rectangle as background
        text_coords = ((text_x, text_y - text_size[1]), (text_x + text_size[0], text_y))
        # Draw text on top of the white rectangle
        cv2.rectangle(annotated_frame, text_coords[0], text_coords[1], Config.background_color, -1)
        cv2.putText(annotated_frame, reversed_text, (text_x, text_y), Config.font, Config.font_scale, Config.text_color, Config.font_thickness)


        text_y -= text_size[1] + 10  # Adjust the y-position for the next line



# Put Default Total Count Text
def defaultTotalCount(annotated_frame, total):
    # Starting position at the bottom of the frame
    text_y = annotated_frame.shape[0] - 200
    
    for i, (key, value) in enumerate(total.items()):
        reversed_text = key + ': ' + str(value)
        new_text_size = cv2.getTextSize(reversed_text, Config.font, Config.font_scale, Config.font_thickness)[0]
        new_text_x = 10  # Center the text horizontally
        new_text_coords = ((new_text_x, text_y - new_text_size[1]), (new_text_x + new_text_size[0], text_y))
        # Draw white rectangle as background
        cv2.rectangle(annotated_frame, new_text_coords[0], new_text_coords[1], Config.background_color, -1)
        # Draw text on top of the white rectangle
        cv2.putText(annotated_frame, reversed_text, (new_text_x, text_y), Config.font, Config.font_scale, Config.text_color, Config.font_thickness)
        # Decrease the y-position for the next line
        text_y -= new_text_size[1] + 10



# Put Saved Total Count Text
def saveTotalCount(annotated_frame):
    count = len(new_arr)
    global new_count
    new_count = new_count + count
    print('Overall Vehicle Count: ',new_count)

    save_count_dict ={
        'Overall Vehicle Count':new_count
    }

    text_y = annotated_frame.shape[0] - 250
    for i, (key, value) in enumerate(save_count_dict.items()):
        reversed_text = key + ': ' + str(value)
        new_text_size = cv2.getTextSize(reversed_text, Config.font, Config.font_scale, Config.font_thickness)[0]
        new_text_x = 10  # Center the text horizontally
        new_text_coords = ((new_text_x, text_y - new_text_size[1]), (new_text_x + new_text_size[0], text_y))
        # Draw white rectangle as background
        cv2.rectangle(annotated_frame, new_text_coords[0], new_text_coords[1], Config.background_color, -1)
        # Draw text on top of the white rectangle
        cv2.putText(annotated_frame, reversed_text, (new_text_x, text_y), Config.font, Config.font_scale, Config.text_color, Config.font_thickness)
        # Decrease the y-position for the next line
        text_y -= new_text_size[1] + 10


