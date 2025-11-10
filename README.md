# 📸 Snap Edit Pro

Advanced computer vision application with image editing, AR filters, real-time gesture control, and emotion detection.

---

## 🚀 How to Run

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Run the Application**

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## ️ Troubleshooting

### **Issue: "ModuleNotFoundError"**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### **Issue: Webcam not detected**
- Check camera permissions in Windows Settings → Privacy → Camera
- Try changing camera index in code (0 to 1)
- Ensure no other application is using the webcam

### **Issue: MediaPipe errors**
```bash
# Reinstall MediaPipe
pip uninstall mediapipe
pip install mediapipe==0.10.13
```

### **Issue: "Port 8501 already in use"**
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

### **Issue: Gesture control not responding**
- Ensure good lighting
- Keep hand within webcam frame
- Pinch fingers closer together
- Check console for debug messages

### **Issue: Music not playing**
```bash
# Reinstall pygame
pip uninstall pygame
pip install pygame
```

---

## 📚 Documentation

For detailed information:
- **README_DETAILED.md** - Complete technical documentation
- **SnapEdit-Pro-Architecture.drawio** - System architecture diagram
