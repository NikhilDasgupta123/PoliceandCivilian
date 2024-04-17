import cv2
from config.config import Config
from utils import putTextPoliceandCivilian
from utils import putTextVehicle


def videoCapture():
   
   while Config.cap.isOpened():
   # Read a frame from the video

      success, frame = Config.cap.read()

      frame = cv2.resize(frame, (900, 700))

      if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = Config.model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        
        
        ### Put Text For Civilian and Police

        # Box Info Police and Civilian
        text_info,total_police_civilin_count = putTextPoliceandCivilian.frameObjectInfo(results)

        
        # Extract detected classes and counts Police and Civilian
        putTextPoliceandCivilian.puttext(annotated_frame,text_info)
        
        # Sum Police and Civilian count (default)  
        putTextPoliceandCivilian.defaultTotalCount(annotated_frame,total_police_civilin_count)
      
        # Save Total People Count
        putTextPoliceandCivilian.saveTotalCount(annotated_frame)





        ### Put Text For Vehicle

        # Box Info Police and Civilian
        text_info,total_four_two_wheel_dict = putTextVehicle.frameObjectInfo(results)

        # Extract detected classes and counts Vehicle
        putTextVehicle.puttext(annotated_frame,text_info)

        # Sum Vehicle count (default)  
        putTextVehicle.defaultTotalCount(annotated_frame,total_four_two_wheel_dict)
      
        # Save Total Vehicle Count
        putTextVehicle.saveTotalCount(annotated_frame)





        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
      else:
        # Break the loop if the end of the video is reached
        break

   # Release the video capture object and close the display window
   Config.cap.release()
   cv2.destroyAllWindows()