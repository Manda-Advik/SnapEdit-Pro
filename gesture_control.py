import cv2
import numpy as np
import mediapipe as mp

class GestureController:
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_hands(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None
    
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
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
    
    def close(self):
        self.hands.close()
