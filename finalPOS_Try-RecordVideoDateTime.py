from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import numpy as np
import time
import pytz
from datetime import datetime
import os

MARKET_SQUARE_VIDEO_PATH = 'people_walking.mp4' #'Sample_Mob.mp4'
model_name = 'yolo'

if model_name == 'detectron':
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    model = DefaultPredictor(cfg)
else:
    model_wt_path = 'yolov8m.pt'
    model = YOLO(model_wt_path)

def process_frame(frame: np.ndarray) -> np.ndarray:
    # detect
    if model_name == 'detectron':
        outputs = model(frame)
        print(outputs["instances"].pred_classes)
        detections = sv.Detections.from_detectron2(outputs)
    else:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.2)]

    # annotate
    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
    frame = box_annotator.annotate(scene=frame, detections=detections)

    return frame, len(detections)

PEOPLE_LIMIT = 29  # Set your people limit
MAX_PEOPLE_LIMIT = 35
SIGNIFICANT_CHANGE = 5  # Number of people change to trigger immediate alert

LAST_ALERT_TIME = 0
PREVIOUS_PEOPLE_COUNT = 0

# Ensure the output directory exists
detection_results_dir = "Detection_Results"
if not os.path.exists(detection_results_dir):
    os.makedirs(detection_results_dir)

# Timezone for Indian Standard Time (IST)
ist = pytz.timezone('Asia/Kolkata')

# Current datetime in IST
current_time_ist = datetime.now(ist)

# Format datetime for the file name
formatted_datetime = current_time_ist.strftime('%Y-%m-%d-%H-%M-%S')

# Model name for file naming
model_name_for_file = 'Detectron' if model_name == 'detectron' else 'YOLO'

# Constructing the file name
file_name = f"Output_Video-{model_name_for_file}-{formatted_datetime}.avi"
out_video_path = os.path.join(detection_results_dir, file_name)


infile = MARKET_SQUARE_VIDEO_PATH
cap = cv2.VideoCapture(infile)

# Prepare VideoWriter
ret, test_frame = cap.read()
if not ret:
    print("Failed to get a frame from the video source.")
    cap.release()
    exit()

frame_height, frame_width = test_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(out_video_path, fourcc, 20.0, (700, 760)) # made changes here

while cv2.waitKey(1) != 27:
    ret, image = cap.read()
    if ret == False:
        break

    frame = cv2.resize(image, (700, 700))

    # Get the dimensions of the original video frame
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Set the size of the white background (adjust as needed)
    padding_bottom = 3  # Number of lines for text at the bottom
    background_width = frame_width
    background_height = frame_height + padding_bottom * 20  # Adjust padding as needed

    # Create a white background
    white_background = np.ones((background_height, background_width, 3), dtype=np.uint8) * 255

    out_frame, people_count = process_frame(image)
    out_frame = cv2.resize(out_frame, (700, 700))

    # Resize the out_frame to match the assigned portion on white_background
    out_frame_resized = cv2.resize(out_frame, (frame_width, frame_height))

    # Create a copy of white_background for each iteration
    white_background_copy = white_background.copy()

    # Place the video frame on the white background
    x_offset = 0  # Left-aligned
    y_offset = 0  # Top-aligned

    # Copy the resized video frame onto the white background
    white_background_copy[y_offset:y_offset + frame_height, x_offset:x_offset + frame_width] = out_frame_resized

    # Display people count and flag information on the white background
    cv2.putText(white_background_copy, f"People Count: {people_count}", (10, frame_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(white_background_copy, f"Max People Limit: {MAX_PEOPLE_LIMIT}", (10, frame_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


    text_position = (10, frame_height + 50)

    alert_text = ''
    if people_count > PEOPLE_LIMIT and people_count <= MAX_PEOPLE_LIMIT:
        alert_text = 'Overcrowding Imminent!'
    elif people_count >= MAX_PEOPLE_LIMIT:
        alert_text = 'OVERCROWDING DETECTED!!!'
    else:
        alert_text = 'Normal'

    if alert_text == "OVERCROWDING DETECTED!!!":
        cv2.putText(white_background_copy, alert_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    elif alert_text == "Overcrowding Imminent!":
        cv2.putText(white_background_copy, alert_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (17, 156, 243))
    else:
        cv2.putText(white_background_copy, alert_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


    # Display the combined frame
    cv2.imshow("Combined_Window", white_background_copy) #Comment This line to avoid xcb error
    out_video.write(white_background_copy) # moved the video writer line here so that the white canvas is also captured in the output video

    # Calculate time since the last alert
    current_time = time.time()
    time_since_last_alert = current_time - LAST_ALERT_TIME

    # Check for a significant change in the number of people
    significant_change = (people_count - PREVIOUS_PEOPLE_COUNT) >= SIGNIFICANT_CHANGE

    # Reset last alert time if people count goes below the limit
    if people_count <= PEOPLE_LIMIT:
        last_alert_time = 0

cap.release()
out_video.release()
print("Video processing completed and the detection results video file is saved to:", out_video_path)
