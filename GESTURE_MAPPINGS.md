# 🎬 Emotion Reactor - Gesture & Video Mappings

## All Available Reactions (7 Videos)

### 1. 😊 **Smiling** → `thumbsup.mp4`
- **Detection**: Mouth corners raised above face center
- **Trigger**: Smile with your mouth (closed or open)
- **Threshold**: Very sensitive (0.04)

### 2. 😐 **Straight Face** → `handsonface.mp4`
- **Detection**: Default state when no other gesture detected
- **Trigger**: Neutral expression, no gestures

### 3. 🙌 **Hands Up** → `handsup.mp4`
- **Detection**: Both hands above shoulders
- **Trigger**: Raise both hands high in the air

### 4. ✌️ **Peace Sign** → `two.mp4`
- **Detection**: Index and middle fingers extended, others folded
- **Trigger**: Make a peace sign / V-sign with your hand
- **Uses**: MediaPipe Hand tracking for finger detection

### 5. 👋 **Waving** → `hello.mp4`
- **Detection**: One hand raised above shoulder
- **Trigger**: Raise one hand up (wave gesture)

### 6. ❤️ **Heart Gesture** → `love.mp4`
- **Detection**: Thumb and index finger forming a heart shape
- **Trigger**: Make a heart shape with your hands
- **Uses**: MediaPipe Hand tracking

### 7. 💪 **Dab** → `dab.mp4`
- **Detection**: Hand in front at chest level near center
- **Trigger**: Do the dab pose (one arm extended, face in elbow)

---

## Detection System

### Technologies Used:
1. **MediaPipe Pose** - Body skeleton tracking (33 landmarks)
2. **MediaPipe Hands** - Hand gesture recognition (21 landmarks per hand)
3. **MediaPipe Face Mesh** - Facial landmark tracking (468 landmarks)

### Priority Order:
1. **Hand Gestures** (Highest priority) - Peace sign, Heart gesture
2. **Body Poses** - Hands up, Waving, Dab
3. **Face Expressions** - Smiling
4. **Default** - Straight face (when nothing else detected)

### Visual Feedback:
- ✅ Green skeleton lines showing body pose
- ✅ Hand landmarks with finger joint connections
- ✅ Face mesh with 468 facial points
- ✅ Real-time state display at top of video

---

## Tips for Best Detection:

### For Hand Gestures:
- **Peace Sign**: Extend only index and middle fingers clearly
- **Heart Gesture**: Keep fingers close but distinct
- Keep hands visible and well-lit

### For Body Poses:
- **Hands Up**: Both hands clearly above shoulder level
- **Waving**: One hand raised, good for quick reactions
- **Dab**: Hand centered in frame at chest level (dab pose)

### For Facial Expression:
- **Smiling**: Natural smile works best, no need to open mouth wide
- Good lighting on face helps
- Face the camera directly

---

## Technical Details:

### Video Specifications:
- All videos are MP4 format
- Located in `assets/cr/` directory
- Auto-loop when playing
- No manual controls (seamless experience)

### Frame Rate:
- Camera feed: ~30 FPS
- Processing: Real-time with MediaPipe
- Video playback: Native browser HTML5 video

### Detection Thresholds:
- **Pose confidence**: 0.5
- **Hand confidence**: 0.5
- **Face confidence**: 0.5
- **Smile threshold**: 0.04 (very sensitive)
- **Finger extension**: 0.02 vertical difference

---

## Testing All Gestures:

1. **Start neutral** → See `handsonface.mp4`
2. **Smile** → Switches to `thumbsup.mp4`
3. **Make peace sign** → Switches to `two.mp4`
4. **Form heart with hands** → Switches to `love.mp4`
5. **Raise both hands** → Switches to `handsup.mp4`
6. **Raise one hand** → Switches to `hello.mp4`
7. **Do the dab** → Switches to `dab.mp4`

Each gesture should trigger the corresponding video to play on the right side!

---

## Future Enhancements:
- Add more hand gestures (thumbs up, OK sign, etc.)
- Emotion intensity levels
- Custom user gestures
- Gesture combinations
- Sound effects for gesture changes
