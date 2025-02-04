#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
import numpy as np
import pyttsx3
import time


# Initialize YOLOv8 model to use GPU

# In[2]:


model = YOLO("yolov8n.pt")
model.overrides["device"] = "cuda"  # Force model to use GPU


# Initialize text-to-speech engine

# In[3]:


engine = pyttsx3.init()


# In[4]:


voices = engine.getProperty('voices')

for voice in voices:
    print(f"Voice Name: {voice.name}, ID: {voice.id}")


# In[5]:


engine.setProperty("rate", 150)  # Set speech rate
engine.setProperty("voice", "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0")


# Define the navigation system

# In[6]:


def navigation_system():
    cap = cv2.VideoCapture(0)  # Access the camera
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    # Variables for speaking navigation commands
    last_spoken_time = time.time()
    current_command = "Move Forward"  # Default command

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab a frame")
                break

            # Convert frame to PyTorch tensor
            frame_tensor = to_tensor(frame).unsqueeze(0).to("cuda")  # Add batch dimension

            # Perform object detection using YOLOv8
            results = model(frame_tensor)
            detected_objects = results[0].boxes  # Access detected objects

            # Initialize variables for navigation
            navigation_command = "Move Forward"
            frame_height, frame_width, _ = frame.shape

            # Calculate segment positions
            segment_width = frame_width // 3

            # Draw the outer rectangle
            cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), (255, 0, 0), 2)

            # Draw the vertical lines dividing the rectangle into three segments
            for i in range(1, 3):
                x = i * segment_width
                cv2.line(frame, (x, 0), (x, frame_height - 1), (255, 0, 0), 2)

            # Extract bounding boxes, labels, and confidences
            boxes = []
            labels = []
            confidence_threshold = 0.5
            for obj in detected_objects:
                if obj.conf >= confidence_threshold:
                    box = obj.xyxy[0].cpu()  # Bounding box coordinates
                    boxes.append(box)
                    labels.append(f"{model.names[int(obj.cls)]} {float(obj.conf):.2f}")

                    # Navigation logic
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2

                    if frame_width * 0.3 <= center_x <= frame_width * 0.7:
                        navigation_command = "Stop"  # Object in the middle
                    elif center_x < frame_width * 0.3:
                        navigation_command = "Turn Left"
                    elif center_x > frame_width * 0.7:
                        navigation_command = "Turn Right"

            # Prepare frame for drawing
            frame_uint8 = torch.from_numpy(frame).permute(2, 0, 1).to(torch.uint8).cpu()  # Convert to CHW and uint8
            if boxes:
                frame_with_boxes = draw_bounding_boxes(
                    frame_uint8,
                    torch.stack(boxes).to(torch.int),
                    labels=labels,
                    colors="green",
                    width=2,
                )
                annotated_frame = to_pil_image(frame_with_boxes)
            else:
                annotated_frame = to_pil_image(frame_uint8)

            # Convert annotated frame back to RGB format for OpenCV
            annotated_frame = np.array(annotated_frame)

            # Overlay navigation command
            cv2.putText(
                annotated_frame,
                f"Command: {navigation_command}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Speak the navigation command every 5 seconds
            current_time = time.time()
            if current_time - last_spoken_time >= 2:
                if current_command != navigation_command:
                    current_command = navigation_command
                engine.say(current_command)
                engine.runAndWait()
                last_spoken_time = current_time

            # Display the annotated frame
            cv2.imshow("Obstacle Avoidance System", annotated_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during processing: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("System shutdown.")


# Run the navigation system

# In[7]:


if __name__ == "__main__":
    navigation_system()


# In[ ]:




