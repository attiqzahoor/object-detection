# Object Detection with YOLOv5 and GUI

This project is a real-time object detection application using YOLOv5 with a graphical interface built in Tkinter. It detects specific everyday objects through your webcam and displays their names below the video feed. The detection is optimized for better accuracy and avoids detecting people.

## 🔍 Features

- 🎥 Real-time webcam detection
- 🧠 YOLOv5m model for accurate results
- 📦 Detects specific objects:
  - **Bottle**
  - **Book**
  - **Cell Phone**
  - **Cup**
- 🖥️ Graphical interface using **Tkinter**
- 🚫 Ignores the "person" class
- ✅ Detects objects only if confidence is above set thresholds

## 🛠 Requirements

- Python 3.8+
- OpenCV
- PyTorch
- PIL (Pillow)
- Tkinter (usually included with Python)

Install dependencies:


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow
🚀 How to Run
Clone the repository and run the script:


git clone https://github.com/attiqzahoor/object-detection.git
cd object-detection
python app.py
Press Q in the video window to quit.

🧠 Model Info
This app uses the yolov5m model from Ultralytics YOLOv5 via torch.hub.load.

👤 Author
Muhammad Attiq
📧 attiqmuhammad51@gmail.com
🐙 GitHub: @attiqzahoor
