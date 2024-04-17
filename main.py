from config.config import Config
from utils.VideoCapture.videoCapture import videoCapture


# main
def main():
    if Config.DEGUG == True:
        print('Debug mode is enabled')
    else:
        print('Debug mode disabled')
    
    # Capture video
    videoCapture()





if __name__ == '__main__':
    main()

#https://github.com/ytl0623/yolov8-fire-car-and-smoke-detection/blob/master/ultralytics/yolo/v8/detect/main.py