import cv2
import numpy as np
import mediapipe as mp
import os
import pygame
from pathlib import Path

class GestureController:
    
    def __init__(self, frames_dir="assets/frames", music_dir="assets/music"):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lower threshold for better detection
            min_tracking_confidence=0.3    # Lower threshold for smoother tracking
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pygame mixer for music
        pygame.mixer.init()
        
        # Load frames and music
        self.frames_dir = frames_dir
        self.music_dir = music_dir
        self.frames = self._load_frames()
        self.music_files = self._load_music()
        
        # Current states
        self.current_frame_index = 0
        self.current_music_index = 0
        self.volume = 50
        
        # Gesture tracking
        self.prev_hand_x = None
        self.prev_hand_y = None
        self.gesture_cooldown = 0
        
        # Play first song
        if self.music_files:
            self._play_music(0)
    
    def _load_frames(self):
        """Load all frame images from the frames directory (including transparent PNGs)"""
        frames = []
        if os.path.exists(self.frames_dir):
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            for frame_file in frame_files:
                frame_path = os.path.join(self.frames_dir, frame_file)
                # Load with alpha channel if PNG
                if frame_file.lower().endswith('.png'):
                    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                else:
                    frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
            print(f"✓ Loaded {len(frames)} frames")
        return frames
    
    def _load_music(self):
        """Load all music files from the music directory"""
        music_files = []
        if os.path.exists(self.music_dir):
            music_files = sorted([os.path.join(self.music_dir, f) for f in os.listdir(self.music_dir) 
                                 if f.lower().endswith(('.mp3', '.wav', '.ogg'))])
            print(f"✓ Loaded {len(music_files)} music files")
        return music_files
    
    def _play_music(self, index):
        """Play music at the specified index"""
        if 0 <= index < len(self.music_files):
            try:
                pygame.mixer.music.load(self.music_files[index])
                pygame.mixer.music.set_volume(self.volume / 100.0)
                pygame.mixer.music.play()
                self.current_music_index = index
                print(f"♪ Playing: {os.path.basename(self.music_files[index])}")
            except Exception as e:
                print(f"Error playing music: {e}")
    
    def detect_hands(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None
    
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_pinching(self, landmarks):
        """Check if thumb and index finger are pinched together - STRICT detection"""
        if not landmarks:
            return False
        
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_ip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        index_pip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        # Calculate distance between thumb tip and index tip
        tip_distance = self.calculate_distance(thumb_tip, index_tip)
        
        # Also check middle joints to ensure proper pinch
        joint_distance = self.calculate_distance(thumb_ip, index_pip)
        
        # Relaxed threshold - easier to detect pinch
        is_pinch = tip_distance < 0.05 and joint_distance < 0.12
        
        return is_pinch
    
    def detect_hand_movement(self, landmarks, h, w):
        """Detect hand movement direction (left/right for song, up/down for volume) - ONLY when pinching"""
        if not landmarks:
            return None
        
        # Check if fingers are pinched - REQUIRED for gesture detection
        if not self.is_pinching(landmarks):
            # Reset tracking when not pinching
            self.prev_hand_x = None
            self.prev_hand_y = None
            return None
        
        # Use wrist position for tracking
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        hand_x = wrist.x
        hand_y = wrist.y
        
        # Decrease cooldown
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return None
        
        movement = None
        
        if self.prev_hand_x is not None:
            # Calculate movement differences
            vertical_diff = self.prev_hand_y - hand_y  # Inverted (up is negative y)
            horizontal_diff = hand_x - self.prev_hand_x
            
            # Lower thresholds for better detection
            vertical_threshold = 0.03
            horizontal_threshold = 0.03  # Made same as vertical for equal sensitivity
            
            # Check which movement is dominant
            if abs(vertical_diff) > vertical_threshold or abs(horizontal_diff) > horizontal_threshold:
                # Determine which direction is more significant (at least 1.5x larger)
                if abs(vertical_diff) > abs(horizontal_diff) * 1.5:
                    # Vertical movement is clearly dominant - volume control
                    if vertical_diff > 0:
                        movement = "UP"
                        print(f"UP detected: vertical_diff={vertical_diff:.3f}, horizontal={horizontal_diff:.3f}")
                    else:
                        movement = "DOWN"
                        print(f"DOWN detected: vertical_diff={vertical_diff:.3f}, horizontal={horizontal_diff:.3f}")
                    self.gesture_cooldown = 2  # Very short cooldown for rapid volume changes
                elif abs(horizontal_diff) > abs(vertical_diff) * 1.5:
                    # Horizontal movement is clearly dominant - change songs/frames
                    if horizontal_diff > 0:
                        movement = "RIGHT"
                        print(f"RIGHT detected: horizontal_diff={horizontal_diff:.3f}, vertical={vertical_diff:.3f}")
                    else:
                        movement = "LEFT"
                        print(f"LEFT detected: horizontal_diff={horizontal_diff:.3f}, vertical={vertical_diff:.3f}")
                    self.gesture_cooldown = 15  # Moderate cooldown to prevent double triggers
                else:
                    # Movement is diagonal or unclear - use the larger one
                    if abs(horizontal_diff) > abs(vertical_diff):
                        if horizontal_diff > 0:
                            movement = "RIGHT"
                            print(f"RIGHT (diagonal) detected: horizontal={horizontal_diff:.3f}")
                        else:
                            movement = "LEFT"
                            print(f"LEFT (diagonal) detected: horizontal={horizontal_diff:.3f}")
                        self.gesture_cooldown = 15
                    else:
                        if vertical_diff > 0:
                            movement = "UP"
                            print(f"UP (diagonal) detected: vertical={vertical_diff:.3f}")
                        else:
                            movement = "DOWN"
                            print(f"DOWN (diagonal) detected: vertical={vertical_diff:.3f}")
                        self.gesture_cooldown = 2
        
        self.prev_hand_x = hand_x
        self.prev_hand_y = hand_y
        
        return movement
    
    def change_frame(self, direction):
        """Change the current frame"""
        if not self.frames:
            return
        
        if direction == "RIGHT":
            self.current_frame_index = (self.current_frame_index + 1) % len(self.frames)
        elif direction == "LEFT":
            self.current_frame_index = (self.current_frame_index - 1) % len(self.frames)
        
        print(f"🖼️ Frame: {self.current_frame_index + 1}/{len(self.frames)}")
    
    def change_music(self, direction):
        """Change the current music track"""
        if not self.music_files:
            return
        
        if direction == "RIGHT":
            next_index = (self.current_music_index + 1) % len(self.music_files)
            self._play_music(next_index)
        elif direction == "LEFT":
            next_index = (self.current_music_index - 1) % len(self.music_files)
            self._play_music(next_index)
    
    def adjust_volume(self, direction):
        """Adjust the music volume"""
        if direction == "UP":
            self.volume = min(100, self.volume + 20)  # Increased from 10 to 20
        elif direction == "DOWN":
            self.volume = max(0, self.volume - 20)  # Increased from 10 to 20
        
        pygame.mixer.music.set_volume(self.volume / 100.0)
        print(f"🔊 Volume: {self.volume}%")
    
    def get_current_frame(self):
        """Get the current frame image"""
        if self.frames:
            return self.frames[self.current_frame_index].copy()
        return None
    
    def process_frame_with_gestures(self, camera_frame, show_text=True):
        """Process camera frame and detect gesture controls"""
        landmarks = self.detect_hands(camera_frame)
        result_image = camera_frame.copy()
        h, w = camera_frame.shape[:2]
        
        # Initialize movement variable
        movement = None
        
        # Draw hand landmarks
        if landmarks:
            self.mp_drawing.draw_landmarks(
                result_image,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Detect hand movement
            movement = self.detect_hand_movement(landmarks, h, w)
            
            if movement:
                if movement in ["LEFT", "RIGHT"]:
                    self.change_frame(movement)
                    self.change_music(movement)
                elif movement in ["UP", "DOWN"]:
                    self.adjust_volume(movement)
        
        # Get current frame overlay
        frame_overlay = self.get_current_frame()
        
        # Only display info on camera frame if show_text is True
        if show_text:
            cv2.putText(result_image, f"Frame: {self.current_frame_index + 1}/{len(self.frames)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(result_image, f"Song: {self.current_music_index + 1}/{len(self.music_files)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(result_image, f"Volume: {self.volume}%", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show pinch indicator - ALWAYS show this (regardless of show_text)
        if landmarks:
            thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tip_distance = self.calculate_distance(thumb_tip, index_tip)
            
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            if self.is_pinching(landmarks):
                # Draw green line and circles when pinching
                cv2.line(result_image, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 4)
                cv2.circle(result_image, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
                cv2.circle(result_image, (index_x, index_y), 10, (0, 255, 0), -1)
                # Show distance in green
                mid_x = (thumb_x + index_x) // 2
                mid_y = (thumb_y + index_y) // 2
                cv2.putText(result_image, f"{tip_distance:.3f}", 
                           (mid_x - 30, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Draw yellow line to show current distance
                cv2.line(result_image, (thumb_x, thumb_y), (index_x, index_y), (255, 255, 0), 2)
                cv2.circle(result_image, (thumb_x, thumb_y), 8, (255, 255, 0), 2)
                cv2.circle(result_image, (index_x, index_y), 8, (255, 255, 0), 2)
                # Show distance in yellow
                mid_x = (thumb_x + index_x) // 2
                mid_y = (thumb_y + index_y) // 2
                cv2.putText(result_image, f"{tip_distance:.3f}", 
                           (mid_x - 30, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show text overlays only if show_text is True
        if show_text and not landmarks:
            cv2.putText(result_image, "NO HAND - Show your hand", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show movement detection with bigger, more visible feedback (only if show_text is True)
        if show_text and movement:
            arrow_map = {
                "LEFT": "◄◄◄ PREVIOUS",
                "RIGHT": "NEXT ►►►",
                "UP": "▲▲▲ VOL UP",
                "DOWN": "▼▼▼ VOL DOWN"
            }
            # Large feedback in center
            cv2.putText(result_image, arrow_map.get(movement, movement), 
                       (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
            # Small feedback at top
            cv2.putText(result_image, f"ACTION: {movement}", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if show_text:
            cv2.putText(result_image, "PINCH and move LEFT/RIGHT: Change song & frame", 
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_image, "PINCH and move UP/DOWN: Adjust volume", 
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image, frame_overlay
    
    def get_volume_from_gesture(self, image):
        landmarks = self.detect_hands(image)
        result_image = image.copy()
        
        if not landmarks:
            return None, result_image
        
        h, w = image.shape[:2]
        
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        distance = self.calculate_distance(thumb_tip, index_tip)
        
        min_distance = 0.02
        max_distance = 0.25
        
        distance = max(min_distance, min(distance, max_distance))
        
        volume = int(((distance - min_distance) / (max_distance - min_distance)) * 100)
        volume = max(0, min(100, volume))
        
        self.mp_drawing.draw_landmarks(
            result_image,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        
        cv2.line(result_image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)
        cv2.circle(result_image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
        cv2.circle(result_image, (index_x, index_y), 10, (255, 0, 0), -1)
        
        mid_x = (thumb_x + index_x) // 2
        mid_y = (thumb_y + index_y) // 2
        cv2.putText(result_image, f"Volume: {volume}%", (mid_x - 50, mid_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return volume, result_image
    
    def detect_gesture_type(self, image):
        landmarks = self.detect_hands(image)
        result_image = image.copy()
        
        if not landmarks:
            return "No Hand", result_image
        
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        index_mcp = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        
        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < middle_mcp.y
        ring_extended = ring_tip.y < ring_mcp.y
        pinky_extended = pinky_tip.y < pinky_mcp.y
        
        gesture = "Unknown"
        
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            gesture = "Peace"
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            gesture = "Pointing"
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            gesture = "Open Hand"
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            gesture = "Fist"
        elif index_extended and pinky_extended and not middle_extended and not ring_extended:
            gesture = "Rock"
        
        self.mp_drawing.draw_landmarks(
            result_image,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        
        h, w = image.shape[:2]
        cv2.putText(result_image, f"Gesture: {gesture}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return gesture, result_image
    
    def run_live_control(self):
        """Run live gesture control with webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n🎮 Gesture Control Started!")
        print("Controls:")
        print("  - Move hand LEFT/RIGHT: Change song & frame")
        print("  - Move hand UP/DOWN: Adjust volume")
        print("  - Press 'q' to quit")
        print("  - Press 'p' to pause/play music")
        print("  - Press 'n' to next song")
        print("  - Press 'm' to previous song\n")
        
        cv2.namedWindow('Gesture Control - Camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Current Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gesture Control - Camera', 640, 480)
        cv2.resizeWindow('Current Frame', 640, 480)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame with gesture detection
            result_frame, frame_overlay = self.process_frame_with_gestures(frame)
            
            # Display camera feed
            cv2.imshow('Gesture Control - Camera', result_frame)
            
            # Display current frame overlay
            if frame_overlay is not None:
                # Resize frame overlay to match display
                frame_display = cv2.resize(frame_overlay, (640, 480))
                cv2.imshow('Current Frame', frame_display)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.pause()
                    print("⏸️ Music paused")
                else:
                    pygame.mixer.music.unpause()
                    print("▶️ Music resumed")
            elif key == ord('n'):
                next_index = (self.current_music_index + 1) % len(self.music_files)
                self._play_music(next_index)
            elif key == ord('m'):
                prev_index = (self.current_music_index - 1) % len(self.music_files)
                self._play_music(prev_index)
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Gesture Control stopped")
    
    def close(self):
        """Clean up resources"""
        self.hands.close()
        pygame.mixer.music.stop()
        pygame.mixer.quit()


# Main function to run gesture control
if __name__ == "__main__":
    controller = GestureController()
    try:
        controller.run_live_control()
    finally:
        controller.close()
