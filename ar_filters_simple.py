import cv2
import numpy as np

class ARFiltersSimple:
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            return faces[0]
        return None
    
    def detect_eyes(self, image, face_rect):
        x, y, w, h = face_rect
        roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        
        absolute_eyes = []
        for (ex, ey, ew, eh) in eyes:
            absolute_eyes.append((x + ex, y + ey, ew, eh))
        
        return absolute_eyes
    
    def apply_sunglasses(self, image):
        """Apply sunglasses overlay"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        x, y, w, h = face
        result = image.copy()
        
        # Detect eyes
        eyes = self.detect_eyes(image, face)
        
        if len(eyes) >= 2:
            # Sort eyes by x coordinate (left to right)
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Get eye centers
            left_eye_x = eyes[0][0] + eyes[0][2] // 2
            left_eye_y = eyes[0][1] + eyes[0][3] // 2
            right_eye_x = eyes[1][0] + eyes[1][2] // 2
            right_eye_y = eyes[1][1] + eyes[1][3] // 2
            
            # Calculate sunglasses dimensions
            eye_distance = int(np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2))
            glasses_width = int(eye_distance * 2.2)
            glasses_height = int(glasses_width * 0.4)
            
            # Center position
            center_x = (left_eye_x + right_eye_x) // 2
            center_y = (left_eye_y + right_eye_y) // 2
            
            # Draw stylized sunglasses
            # Left lens
            left_center = (center_x - eye_distance // 2, center_y)
            cv2.ellipse(result, left_center, (glasses_width // 4, glasses_height // 2), 
                       0, 0, 360, (0, 0, 0), -1)
            cv2.ellipse(result, left_center, (glasses_width // 4, glasses_height // 2), 
                       0, 0, 360, (50, 50, 50), 3)
            
            # Right lens
            right_center = (center_x + eye_distance // 2, center_y)
            cv2.ellipse(result, right_center, (glasses_width // 4, glasses_height // 2), 
                       0, 0, 360, (0, 0, 0), -1)
            cv2.ellipse(result, right_center, (glasses_width // 4, glasses_height // 2), 
                       0, 0, 360, (50, 50, 50), 3)
            
            # Bridge
            cv2.line(result, left_center, right_center, (50, 50, 50), 3)
        else:
            # Fallback: draw sunglasses based on face position
            center_x = x + w // 2
            center_y = y + h // 3
            glasses_width = w // 3
            
            # Left lens
            cv2.ellipse(result, (center_x - w//6, center_y), (glasses_width//2, glasses_width//3), 
                       0, 0, 360, (0, 0, 0), -1)
            cv2.ellipse(result, (center_x - w//6, center_y), (glasses_width//2, glasses_width//3), 
                       0, 0, 360, (50, 50, 50), 3)
            
            # Right lens
            cv2.ellipse(result, (center_x + w//6, center_y), (glasses_width//2, glasses_width//3), 
                       0, 0, 360, (0, 0, 0), -1)
            cv2.ellipse(result, (center_x + w//6, center_y), (glasses_width//2, glasses_width//3), 
                       0, 0, 360, (50, 50, 50), 3)
        
        return result
    
    def apply_hat(self, image):
        """Apply hat overlay"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        x, y, w, h = face
        result = image.copy()
        
        # Hat position (above the face)
        hat_width = int(w * 1.5)
        hat_height = int(hat_width * 0.6)
        hat_x = x + w // 2
        hat_y = y - hat_height // 4
        
        # Draw a simple top hat
        # Hat brim
        brim_top_left = (hat_x - hat_width // 2, hat_y)
        brim_bottom_right = (hat_x + hat_width // 2, hat_y + hat_height // 4)
        cv2.rectangle(result, brim_top_left, brim_bottom_right, (0, 0, 0), -1)
        cv2.rectangle(result, brim_top_left, brim_bottom_right, (100, 100, 100), 2)
        
        # Hat top
        top_left = (hat_x - hat_width // 3, hat_y - hat_height)
        top_right = (hat_x + hat_width // 3, hat_y)
        cv2.rectangle(result, top_left, top_right, (0, 0, 0), -1)
        cv2.rectangle(result, top_left, top_right, (100, 100, 100), 2)
        
        return result
    
    def apply_dog_filter(self, image):
        """Apply dog ears and nose"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        x, y, w, h = face
        result = image.copy()
        
        # Dog nose position (center bottom of face)
        nose_x = x + w // 2
        nose_y = y + int(h * 0.7)
        nose_size = w // 8
        
        # Draw dog nose
        cv2.ellipse(result, (nose_x, nose_y), (nose_size, nose_size // 2), 
                   0, 0, 360, (0, 0, 0), -1)
        
        # Draw dog ears (triangles on sides)
        ear_size = w // 3
        
        # Left ear
        left_ear_pts = np.array([
            [x - ear_size // 2, y],
            [x, y + ear_size],
            [x + ear_size // 2, y]
        ], np.int32)
        cv2.fillPoly(result, [left_ear_pts], (139, 90, 43))
        cv2.polylines(result, [left_ear_pts], True, (100, 60, 30), 2)
        
        # Right ear
        right_ear_pts = np.array([
            [x + w - ear_size // 2, y],
            [x + w, y + ear_size],
            [x + w + ear_size // 2, y]
        ], np.int32)
        cv2.fillPoly(result, [right_ear_pts], (139, 90, 43))
        cv2.polylines(result, [right_ear_pts], True, (100, 60, 30), 2)
        
        return result
    
    def apply_crown(self, image):
        """Apply crown overlay"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        x, y, w, h = face
        result = image.copy()
        
        # Crown position (top of head)
        crown_width = w
        crown_height = int(crown_width * 0.4)
        crown_x = x + w // 2
        crown_y = y - crown_height // 2
        
        # Draw crown points
        num_points = 5
        points = []
        base_y = crown_y + crown_height // 3
        
        for i in range(num_points * 2 + 1):
            px = x + (crown_width * i) // (num_points * 2)
            if i % 2 == 0:
                py = crown_y
            else:
                py = base_y
            points.append([px, py])
        
        # Add base points
        points.append([x + crown_width, base_y])
        points.append([x, base_y])
        
        crown_pts = np.array(points, np.int32)
        cv2.fillPoly(result, [crown_pts], (0, 215, 255))  # Gold color
        cv2.polylines(result, [crown_pts], True, (0, 165, 255), 2)
        
        # Add jewels
        for i in range(0, num_points * 2 + 1, 2):
            px = x + (crown_width * i) // (num_points * 2)
            py = crown_y + 10
            cv2.circle(result, (px, py), 5, (0, 0, 255), -1)
        
        return result
    
    def apply_heart_eyes(self, image):
        """Apply heart eyes effect"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        x, y, w, h = face
        result = image.copy()
        
        # Detect eyes
        eyes = self.detect_eyes(image, face)
        
        heart_size = w // 8
        
        if len(eyes) >= 2:
            # Sort eyes by x coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Draw hearts over eyes
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_x = ex + ew // 2
                eye_y = ey + eh // 2
                
                # Create heart shape
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
        else:
            # Fallback: draw hearts at estimated eye positions
            left_eye_x = x + w // 3
            right_eye_x = x + 2 * w // 3
            eye_y = y + h // 3
            
            for eye_x in [left_eye_x, right_eye_x]:
                cv2.circle(result, (eye_x - heart_size // 4, eye_y - heart_size // 4), 
                          heart_size // 2, (0, 0, 255), -1)
                cv2.circle(result, (eye_x + heart_size // 4, eye_y - heart_size // 4), 
                          heart_size // 2, (0, 0, 255), -1)
                
                heart_pts = np.array([
                    [eye_x - heart_size // 2, eye_y - heart_size // 4],
                    [eye_x + heart_size // 2, eye_y - heart_size // 4],
                    [eye_x, eye_y + heart_size // 2]
                ], np.int32)
                cv2.fillPoly(result, [heart_pts], (0, 0, 255))
        
        return result
    
    def apply_sparkles(self, image):
        """Apply sparkle effects around face"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        x, y, w, h = face
        result = image.copy()
        
        # Add sparkles around face perimeter
        sparkle_positions = [
            (x, y),  # Top left
            (x + w // 2, y - 20),  # Top center
            (x + w, y),  # Top right
            (x - 20, y + h // 2),  # Left center
            (x + w + 20, y + h // 2),  # Right center
            (x, y + h),  # Bottom left
            (x + w, y + h),  # Bottom right
            (x + w // 4, y - 10),  # Extra sparkles
            (x + 3 * w // 4, y - 10),
        ]
        
        for (sx, sy) in sparkle_positions:
            sparkle_size = 15
            color = (255, 255, 0)  # Yellow
            
            # Draw sparkle (star shape)
            cv2.line(result, (sx - sparkle_size, sy), (sx + sparkle_size, sy), color, 2)
            cv2.line(result, (sx, sy - sparkle_size), (sx, sy + sparkle_size), color, 2)
            
            # Draw X
            offset = int(sparkle_size * 0.7)
            cv2.line(result, (sx - offset, sy - offset), (sx + offset, sy + offset), color, 2)
            cv2.line(result, (sx - offset, sy + offset), (sx + offset, sy - offset), color, 2)
        
        return result
    
    def close(self):
        """Clean up resources (not needed for OpenCV)"""
        pass
