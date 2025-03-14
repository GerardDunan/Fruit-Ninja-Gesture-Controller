
# Fruit Ninja Gesture Controller
![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.9+-orange.svg)

A secure hand gesture-based controller for Fruit Ninja that uses face recognition and hand tracking to create a touchless gaming experience.

## ✨ Features
- **👤 Facial Recognition Security**: Only registered users can control the game
- **🔐 Two-Step Authentication**: Register with peace sign ✌️ followed by fist ✊
- **👆 Intuitive Control**: Slice fruit with natural hand movements
- **🎮 Visual Feedback**: See your slicing trail and authentication status
- **💾 Persistent Registration**: Face image saved between sessions

## 📋 Requirements
- Python 3.6+
- Webcam
- Libraries:
  - OpenCV
  - PyAutoGUI
  - NumPy
  - MediaPipe
- BlueStacks (to run Fruit Ninja on PC)
- Fruit Ninja app from Google Play Store

## 🚀 Installation

### Setting up the Controller
```bash
# Clone the repository
git clone https://github.com/GerardDunan/Fruit-Ninja-Gesture-Controller.git
cd fruitninja-controller

# Install required packages
pip install -r requirements.txt
```

### Setting up Fruit Ninja
1. Download and install [BlueStacks](https://www.bluestacks.com/)
2. Launch BlueStacks and sign in with your Google account
3. Open Google Play Store within BlueStacks
4. Search for "Fruit Ninja" and install it
5. Launch Fruit Ninja from your BlueStacks home screen

## 🎮 Usage
### Initial Setup
1. Launch Fruit Ninja in BlueStacks
2. Run the controller script:
   ```bash
   python main.py
   ```
3. Position yourself in front of the webcam
4. Make a peace sign ✌️ and hold until the progress bar fills
5. When prompted, make a fist ✊ until the second progress bar fills
6. Your face is now registered and saved

### Playing the Game
1. Make sure Fruit Ninja is open and active in BlueStacks
2. Position yourself in front of the webcam
3. Extend your index finger (pointing) to control the cursor
4. Move your finger to slice fruit - the mouse button is automatically held down

### Security Reset
- Press `r` to reset face registration (allows a new user to register)
- Press `q` to quit the application

## 🔍 How It Works
The application uses:
- **MediaPipe** for real-time hand gesture and face detection
- **OpenCV** for image processing and visual feedback
- **PyAutoGUI** to control mouse movements based on finger position
- **BlueStacks** to run the Android version of Fruit Ninja on PC
Hand gestures are detected frame-by-frame and translated to on-screen actions.

## 🔒 Security Features
- Face recognition ensures only the registered user can play
- Two-step gesture authentication prevents accidental registration
- Registered face is saved as an image for visual confirmation
- Red warning indicators when an unregistered user attempts to play

## 🔧 Troubleshooting
- **Face Not Recognized**: Try moving closer to the camera or in better lighting
- **Gestures Not Detected**: Ensure your hands are fully visible in the frame
- **Mouse Issues**: Press 'q' to quit cleanly if the mouse gets stuck
- **Registration Problems**: Press 'r' to reset completely and try registration again
- **BlueStacks Performance**: Close other applications to ensure smooth gameplay

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- [MediaPipe](https://mediapipe.dev/) for the hand tracking and face detection models
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse control
- [BlueStacks](https://www.bluestacks.com/) for Android emulation
---
Created with ❤️ by **Gerard**
