import cv2


# Function to convert Ultralytics bounding boxes to OpenCV format
def ultralytics_to_cv_boxes(boxes):
    # Format: [x1, y1, x2, y2]
    return [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in boxes]




# Draw bounding boxes
def drawingBoxes(results,frame):
    for pred in results.pred:
        for det in pred:
            box = ultralytics_to_cv_boxes([det[0:4]])
            cv2.rectangle(frame, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (0, 255, 0), 2)



    