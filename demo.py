from ultralytics import YOLO
import cv2
import numpy as np




# def reverse_text(text):
#     return text[::-1]



# Load the YOLOv8 model
model = YOLO('WEIGHT/last(4).pt')

model2 = YOLO('WEIGHT/model.pt')

model.model.load_state_dict(model2.model.state_dict())

# Add the weights from the additional model to the original model
# model.model.load_model_weights(model2.model)

# Open the video file
video_path = "Video/41.mp4"
cap = cv2.VideoCapture(video_path)

class_names = ['dog','person','cat','tv','car','meatballs','marinara sauce','tomato soup','chicken noodle soup','french onion soup','chicken breast','ribs',
'pulled pork','hamburger','cavity','Police''Civilian']

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()



        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            probs = result.probs  # Class probabilities for classification outputs
            cls = boxes.cls.tolist()  # Convert tensor to list
            xyxy = boxes.xyxy
            xywh = boxes.xywh  # box with xywh format, (N, 4)
            conf = boxes.conf
            print(cls)
            for class_index in cls:
                if class_index == 15.0 or class_index==0:
                    global police_count
                    police_count = cls.count(15.0)
                    # print('Police count',cls.count(15.0))
                elif class_index == 16.0 or class_index==0:
                    global civilian_count
                    civilian_count = cls.count(16.0)
                    # print('Civilian count',cls.count(15.0))
                # elif class_index!=15.0:
                #     police_count=0
                # elif class_index!=16.0:
                #     civilian_count=0
                    
            
        # print('Total Police Count: ',police_count)
        # print('Total Civilian Count: ',civilian_count)
                

        # Add text to the annotated frame with white background
        # text = "YOLOv8 Tracking"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.8
        # font_thickness = 2
        # text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        # text_x = 20
        # text_y = annotated_frame.shape[0] - 20
        # text_color = (0, 0, 0)  # black text color
        # bg_color = (255, 255, 255)  # white background color
        # bg_padding = 5
        # org = (00, 185) 
        # cv2.rectangle(annotated_frame, (text_x - bg_padding, text_y - text_size[1] - bg_padding),
        #               (text_x + text_size[0] + bg_padding, text_y + bg_padding), bg_color, -1)
        
        
        

        

        # Text
        text_info = {
            'Total Police Count' : police_count,
            'Total Civilian Count' : civilian_count
        }

        # Define the font parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Reduced font size
        text_color = (0, 0, 0)  # Black text color
        background_color = (255, 255, 255)  # White background color
        font_thickness = 2

        # Determine the total height of the text
        # total_text_height = sum(cv2.getTextSize(key + ': ' + str(value), font, font_scale, font_thickness)[0][1] for key, value in text_info.items())


        # Display the text upside down at the bottom of the frame
        text_y = annotated_frame.shape[0] - 10  # Starting position at the bottom of the frame
        
        
        # Reverse and display the text upside down
        for i, (key, value) in enumerate(text_info.items()):
            reversed_text = key + ': ' + str(value)
            text_size = cv2.getTextSize(reversed_text, font, font_scale, font_thickness)[0]
            text_x = (annotated_frame.shape[1] - text_size[0]) // 2  # Center the text horizontally
            # Draw white rectangle as background
            text_coords = ((text_x, text_y - text_size[1]), (text_x + text_size[0], text_y))
            # Draw text on top of the white rectangle
            cv2.rectangle(annotated_frame, text_coords[0], text_coords[1], background_color, -1)
            cv2.putText(annotated_frame, reversed_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

            text_y -= text_size[1] + 10  # Adjust the y-position for the next line

        
        
        
        
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
