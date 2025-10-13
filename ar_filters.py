"""
AR Filters implementation using MediaPipe face detection
"""
import cv2
import numpy as np
import mediapipe as mp


class ARFilters:
    """Handles AR filter application with face detection"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
    
    def detect_face(self, image):
        """Detect face in image and return landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None
    
    def apply_sunglasses(self, image):
        """Apply sunglasses overlay"""
        landmarks = self.detect_face(image)
        if not landmarks:
            return image
        
        h, w = image.shape[:2]
        result = image.copy()
        
        # Get eye landmarks (approximate positions)
        left_eye = landmarks.landmark[33]  # Left eye outer corner
        right_eye = landmarks.landmark[263]  # Right eye outer corner
        
        left_x, left_y = int(left_eye.x * w), int(left_eye.y * h)
        right_x, right_y = int(right_eye.x * w), int(right_eye.y * h)
        
        # Calculate sunglasses dimensions
        eye_distance = int(np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2))
        glasses_width = int(eye_distance * 1.8)
        glasses_height = int(glasses_width * 0.4)
        
        # Center position
        center_x = (left_x + right_x) // 2
        center_y = (left_y + right_y) // 2
        
        # Draw stylized sunglasses
        # Left lens
        left_center = (center_x - eye_distance // 3, center_y)
        cv2.ellipse(result, left_center, (glasses_width // 4, glasses_height // 2), 
                   0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(result, left_center, (glasses_width // 4, glasses_height // 2), 
                   0, 0, 360, (50, 50, 50), 3)
        
        # Right lens
        right_center = (center_x + eye_distance // 3, center_y)
        cv2.ellipse(result, right_center, (glasses_width // 4, glasses_height // 2), 
                   0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(result, right_center, (glasses_width // 4, glasses_height // 2), 
                   0, 0, 360, (50, 50, 50), 3)
        
        # Bridge
        cv2.line(result, left_center, right_center, (50, 50, 50), 3)
        
        return result
    
    def apply_hat(self, image):
        """Apply hat overlay"""
        landmarks = self.detect_face(image)
        if not landmarks:
            return image
        
        h, w = image.shape[:2]
        result = image.copy()
        
        # Get forehead position
        forehead = landmarks.landmark[10]  # Top of face
        left_temple = landmarks.landmark[21]
        right_temple = landmarks.landmark[251]
        
        forehead_x, forehead_y = int(forehead.x * w), int(forehead.y * h)
        left_x = int(left_temple.x * w)
        right_x = int(right_temple.x * w)
        
        # Calculate hat dimensions
        hat_width = int(abs(right_x - left_x) * 1.5)
        hat_height = int(hat_width * 0.6)
        
        # Draw a simple top hat
        # Hat brim
        brim_top_left = (forehead_x - hat_width // 2, forehead_y - hat_height // 4)
        brim_bottom_right = (forehead_x + hat_width // 2, forehead_y)
        cv2.rectangle(result, brim_top_left, brim_bottom_right, (0, 0, 0), -1)
        cv2.rectangle(result, brim_top_left, brim_bottom_right, (100, 100, 100), 2)
        
        # Hat top
        top_left = (forehead_x - hat_width // 3, forehead_y - hat_height)
        top_right = (forehead_x + hat_width // 3, forehead_y - hat_height // 4)
        cv2.rectangle(result, top_left, top_right, (0, 0, 0), -1)
        cv2.rectangle(result, top_left, top_right, (100, 100, 100), 2)
        
        return result
    
    def apply_dog_filter(self, image):
        """Apply dog ears and nose"""
        landmarks = self.detect_face(image)
        if not landmarks:
            return image
        
        h, w = image.shape[:2]
        result = image.copy()
        
        # Get key points
        nose_tip = landmarks.landmark[4]
        left_ear = landmarks.landmark[127]
        right_ear = landmarks.landmark[356]
        
        nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
        left_ear_x, left_ear_y = int(left_ear.x * w), int(left_ear.y * h)
        right_ear_x, right_ear_y = int(right_ear.x * w), int(right_ear.y * h)
        
        # Draw dog nose
        nose_size = 20
        cv2.ellipse(result, (nose_x, nose_y), (nose_size, nose_size // 2), 
                   0, 0, 360, (0, 0, 0), -1)
        
        # Draw dog ears (triangles)
        ear_size = 40
        
        # Left ear
        left_ear_pts = np.array([
            [left_ear_x - ear_size, left_ear_y - ear_size * 2],
            [left_ear_x - ear_size // 2, left_ear_y],
            [left_ear_x + ear_size // 2, left_ear_y]
        ], np.int32)
        cv2.fillPoly(result, [left_ear_pts], (139, 90, 43))
        cv2.polylines(result, [left_ear_pts], True, (100, 60, 30), 2)
        
        # Right ear
        right_ear_pts = np.array([
            [right_ear_x + ear_size, right_ear_y - ear_size * 2],
            [right_ear_x + ear_size // 2, right_ear_y],
            [right_ear_x - ear_size // 2, right_ear_y]
        ], np.int32)
        cv2.fillPoly(result, [right_ear_pts], (139, 90, 43))
        cv2.polylines(result, [right_ear_pts], True, (100, 60, 30), 2)
        
        return result
    
    def apply_crown(self, image):
        """Apply crown overlay"""
        landmarks = self.detect_face(image)
        if not landmarks:
            return image
        
        h, w = image.shape[:2]
        result = image.copy()
        
        # Get top of head
        top = landmarks.landmark[10]
        left = landmarks.landmark[21]
        right = landmarks.landmark[251]
        
        top_x, top_y = int(top.x * w), int(top.y * h)
        left_x = int(left.x * w)
        right_x = int(right.x * w)
        
        crown_width = abs(right_x - left_x)
        crown_height = int(crown_width * 0.4)
        
        # Draw crown points
        num_points = 5
        points = []
        base_y = top_y - crown_height // 3
        
        for i in range(num_points * 2 + 1):
            x = left_x + (crown_width * i) // (num_points * 2)
            if i % 2 == 0:
                y = top_y - crown_height
            else:
                y = base_y
            points.append([x, y])
        
        # Add base points
        points.append([right_x, base_y])
        points.append([left_x, base_y])
        
        crown_pts = np.array(points, np.int32)
        cv2.fillPoly(result, [crown_pts], (0, 215, 255))  # Gold color
        cv2.polylines(result, [crown_pts], True, (0, 165, 255), 2)
        
        # Add jewels
        for i in range(0, num_points * 2 + 1, 2):
            x = left_x + (crown_width * i) // (num_points * 2)
            y = top_y - crown_height + 10
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
        
        return result
    
    def apply_heart_eyes(self, image):
        """Apply heart eyes effect"""
        landmarks = self.detect_face(image)
        if not landmarks:
            return image
        
        h, w = image.shape[:2]
        result = image.copy()
        
        # Get eye positions
        left_eye = landmarks.landmark[159]  # Left eye center
        right_eye = landmarks.landmark[386]  # Right eye center
        
        left_x, left_y = int(left_eye.x * w), int(left_eye.y * h)
        right_x, right_y = int(right_eye.x * w), int(right_eye.y * h)
        
        heart_size = 25
        
        # Draw hearts over eyes
        for eye_x, eye_y in [(left_x, left_y), (right_x, right_y)]:
            # Create heart shape using circles and triangle
            cv2.circle(result, (eye_x - heart_size // 4, eye_y - heart_size // 4), 
                      heart_size // 2, (0, 0, 255), -1)
            cv2.circle(result, (eye_x + heart_size // 4, eye_y - heart_size // 4), 
                      heart_size // 2, (0, 0, 255), -1)
            
            # Bottom triangle
            heart_pts = np.array([
                [eye_x - heart_size // 2, eye_y - heart_size // 4],
                [eye_x + heart_size // 2, eye_y - heart_size // 4],
                [eye_x, eye_y + heart_size // 2]
            ], np.int32)
            cv2.fillPoly(result, [heart_pts], (0, 0, 255))
        
        return result
    
    def apply_sparkles(self, image):
        """Apply sparkle effects around face"""
        landmarks = self.detect_face(image)
        if not landmarks:
            return image
        
        h, w = image.shape[:2]
        result = image.copy()
        
        # Get face boundary points
        face_points = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 
                      288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 
                      150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Add sparkles around face
        for idx in face_points[::3]:  # Every 3rd point
            point = landmarks.landmark[idx]
            x, y = int(point.x * w), int(point.y * h)
            
            # Draw sparkle (star shape)
            sparkle_size = 15
            color = (255, 255, 0)  # Yellow
            
            # Draw cross
            cv2.line(result, (x - sparkle_size, y), (x + sparkle_size, y), color, 2)
            cv2.line(result, (x, y - sparkle_size), (x, y + sparkle_size), color, 2)
            
            # Draw X
            offset = int(sparkle_size * 0.7)
            cv2.line(result, (x - offset, y - offset), (x + offset, y + offset), color, 2)
            cv2.line(result, (x - offset, y + offset), (x + offset, y - offset), color, 2)
        
        return result
    
    def close(self):
        """Clean up resources"""
        self.face_mesh.close()
        self.face_detection.close()
