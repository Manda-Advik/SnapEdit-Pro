# 📸 Interactive AR Photo Editor

An innovative photo editing system that combines traditional image editing with modern computer vision and augmented reality features. Built for the **IKT213** course project.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)

## 🌟 Features

### ✂️ Basic Editing Tools
- **Transform Operations**: Crop, resize, rotate, and flip images
- **Adjustments**: Brightness and contrast control
- **Filters**: Grayscale, Sepia, Blur, Sharpen, Edge Detection, Vintage, Cool, and Warm filters

### 😎 AR Filters
Real-time face detection with fun overlays:
- 😎 **Sunglasses**: Cool shades overlay
- 🎩 **Hat**: Top hat accessory
- 🐶 **Dog Filter**: Cute dog ears and nose
- 👑 **Crown**: Royal crown overlay
- 😍 **Heart Eyes**: Romantic heart-shaped eyes
- ✨ **Sparkles**: Magical sparkle effects around the face

### ✋ Gesture-Based Music Control
- Real-time hand tracking using MediaPipe
- Control background music volume by pinching fingers
- Gesture recognition (Peace, Pointing, Open Hand, Fist, Rock)
- Hands-free, intuitive interaction

## 🛠️ Tech Stack

- **Python 3.8+**: Core programming language
- **OpenCV**: Image processing and computer vision
- **MediaPipe**: Face and hand landmark detection
- **Streamlit**: Interactive web UI
- **Pygame**: Background music control
- **NumPy**: Numerical operations
- **Pillow**: Image manipulation

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (optional, for real-time capture)
- Windows/Linux/macOS

## 🚀 Installation

### 1. Clone or Download the Repository

```bash
cd windsurf-project-3
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Upload an Image**
   - Click "Browse files" in the sidebar
   - Or use the "Use Sample Image" button for testing

2. **Select a Mode**
   - **Basic Editing**: Traditional photo editing tools
   - **AR Filters**: Face detection with fun overlays
   - **Gesture Control**: Hand gesture recognition for music control

3. **Apply Edits**
   - Use the various tools and filters
   - See real-time preview of changes
   - Download the edited image when done

### Basic Editing Mode

- **Transform Tab**: Resize, rotate, and flip operations
- **Adjust Tab**: Brightness and contrast adjustments
- **Filters Tab**: Apply various artistic filters
- **Reset Tab**: Reset to original or save current state

### AR Filters Mode

- Click on any filter button to apply face overlays
- Filters automatically detect faces in the image
- Remove filters or save the result

### Gesture Control Mode

- Upload an image showing your hand
- Click "Analyze Hand Gesture" to detect volume from finger distance
- Use "Identify Gesture" to recognize hand gestures
- Control music volume with hand gestures

## 📁 Project Structure

```
windsurf-project-3/
│
├── app.py                  # Main Streamlit application
├── image_editor.py         # Basic image editing functions
├── ar_filters.py          # AR filter implementations
├── gesture_control.py     # Hand gesture recognition
├── music_controller.py    # Background music control
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
└── music/                # (Optional) Music files directory
    ├── cool_vibes.mp3
    ├── jazz.mp3
    ├── playful.mp3
    ├── royal.mp3
    ├── romantic.mp3
    └── magical.mp3
```

## 🎵 Adding Background Music (Optional)

To enable filter-specific background music:

1. Create a `music` directory in the project root
2. Add MP3 files with the following names:
   - `cool_vibes.mp3` - For Sunglasses filter
   - `jazz.mp3` - For Hat filter
   - `playful.mp3` - For Dog filter
   - `royal.mp3` - For Crown filter
   - `romantic.mp3` - For Heart Eyes filter
   - `magical.mp3` - For Sparkles filter
   - `default.mp3` - Default background music

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'cv2'`
- **Solution**: Install OpenCV: `pip install opencv-python`

**Issue**: MediaPipe not working
- **Solution**: Ensure you have the correct version: `pip install mediapipe==0.10.8`

**Issue**: Streamlit not opening in browser
- **Solution**: Manually navigate to `http://localhost:8501`

**Issue**: Face/hand detection not working
- **Solution**: Ensure good lighting and clear visibility of face/hand in the image

## 🎯 Key Concepts Demonstrated

1. **Image Processing**: Using OpenCV for various image transformations
2. **Computer Vision**: Face and hand landmark detection with MediaPipe
3. **Augmented Reality**: Overlay graphics based on facial features
4. **Gesture Recognition**: Interpreting hand gestures for control
5. **UI/UX Design**: Creating an intuitive interface with Streamlit
6. **Multimedia Integration**: Combining image, audio, and interactive elements

## 🚀 Future Enhancements

- [ ] Real-time webcam support for live AR filters
- [ ] More AR filters (cat ears, makeup, masks)
- [ ] Video editing capabilities
- [ ] Custom filter creation tool
- [ ] Social media sharing integration
- [ ] Batch processing for multiple images
- [ ] Advanced gesture controls (zoom, rotate with gestures)
- [ ] Machine learning-based filter suggestions

## 📝 Course Information

**Course**: IKT213  
**Project**: Interactive Photo Editing System with AR and Gesture Control  
**Focus**: Computer Vision, Image Processing, Augmented Reality

## 🤝 Contributing

This is a course project, but suggestions and improvements are welcome!

## 📄 License

This project is created for educational purposes as part of the IKT213 course.

## 👥 Credits

- **OpenCV**: Computer vision library
- **MediaPipe**: ML solutions for face and hand tracking
- **Streamlit**: Web app framework
- **Pygame**: Multimedia library for audio

## 📧 Support

For issues or questions related to this project, please refer to the course materials or contact the instructor.

---

**Built with ❤️ for IKT213 - Demonstrating the power of Computer Vision and AI in multimedia applications**
