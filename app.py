import torch
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

# Load YOLOv5 medium model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# Allowed objects with individual confidence thresholds
thresholds = {
    'bottle': 0.50,
    'book': 0.50,
    'cell phone': 0.70,
    'cup': 0.50
}

# Tkinter setup
root = Tk()
root.title("YOLOv5 Object Detection")

video_label = Label(root)
video_label.pack()

object_names = StringVar()
object_label = Label(root, textvariable=object_names, font=("Arial", 14))
object_label.pack()

# Open webcam
cap = cv2.VideoCapture(0)

def detect_and_display():
    ret, frame = cap.read()
    if not ret:
        root.after(10, detect_and_display)
        return

    # Run detection
    results = model(frame)
    df = results.pandas().xyxy[0]

    # Check and filter results
    if not df.empty and 'name' in df.columns:
        filtered_df = df[df.apply(
            lambda row: row['name'] in thresholds and row['confidence'] >= thresholds[row['name']],
            axis=1
        )]
    else:
        filtered_df = df  # empty

    # Draw bounding boxes
    for _, row in filtered_df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = row['name']
        confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Update GUI with detected object names
    if not filtered_df.empty:
        detected = ", ".join(filtered_df['name'].unique())
        object_names.set("Detected: " + detected)
    else:
        object_names.set("Detected: None")

    # Convert image to RGB and update GUI
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, detect_and_display)

# Start detection
detect_and_display()
root.mainloop()

# Cleanup on exit
cap.release()
cv2.destroyAllWindows()
