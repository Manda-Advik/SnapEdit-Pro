"""
Custom AR Filters using image overlays
Allows users to provide their own sunglasses, hats, and other AR filter images
"""
import cv2
import numpy as np
import os


class ARFiltersCustom:
    """Handles AR filter application with custom image overlays"""
    
    def __init__(self, assets_dir="assets/ar_filters"):
        """
        Initialize AR Filters with custom assets
        
        Args:
            assets_dir: Directory containing AR filter images
        """
        self.assets_dir = assets_dir
        
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Load custom overlay images (with alpha channel for transparency)
        self.overlays = {}
        self._load_overlays()
    
    def _load_overlays(self):
        """Load all overlay images from assets directory dynamically"""
        # Automatically load all PNG files from the assets directory
        if not os.path.exists(self.assets_dir):
            print(f"⚠️ Assets directory not found: {self.assets_dir}")
            return
        
        for filename in os.listdir(self.assets_dir):
            if filename.lower().endswith('.png'):
                # Create key from filename (remove .png extension)
                key = filename[:-4].replace('-', '_').replace(' ', '_').lower()
                filepath = os.path.join(self.assets_dir, filename)
                
                # Load with alpha channel (UNCHANGED flag preserves transparency)
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.overlays[key] = img
                    print(f"✓ Loaded {key} overlay: {filename}")
                else:
                    print(f"✗ Failed to load {key} overlay: {filename}")
        
        print(f"📦 Total filters loaded: {len(self.overlays)}")
    
    def detect_face(self, image):
        """Detect face in image and return face rectangle"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            return faces[0]  # Return first face (x, y, w, h)
        return None
    
    def detect_eyes(self, image, face_rect):
        """Detect eyes within face region"""
        x, y, w, h = face_rect
        roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        
        # Convert eye coordinates to absolute image coordinates
        absolute_eyes = []
        for (ex, ey, ew, eh) in eyes:
            absolute_eyes.append((x + ex, y + ey, ew, eh))
        
        return absolute_eyes
    
    def overlay_image(self, background, overlay, x, y, width, height):
        """
        Overlay an image with transparency onto background
        
        Args:
            background: Background image
            overlay: Overlay image (with alpha channel)
            x, y: Position to place overlay
            width, height: Size to resize overlay to
        """
        # Resize overlay to target dimensions
        overlay_resized = cv2.resize(overlay, (width, height))
        
        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        
        if x + width > background.shape[1]:
            width = background.shape[1] - x
            overlay_resized = cv2.resize(overlay, (width, height))
        
        if y + height > background.shape[0]:
            height = background.shape[0] - y
            overlay_resized = cv2.resize(overlay, (width, height))
        
        # Split overlay into color and alpha channels
        if overlay_resized.shape[2] == 4:  # Has alpha channel
            overlay_rgb = overlay_resized[:, :, :3]
            overlay_alpha = overlay_resized[:, :, 3] / 255.0
            
            # Get the region of interest
            roi = background[y:y+height, x:x+width]
            
            # Blend images
            for c in range(3):
                roi[:, :, c] = (overlay_alpha * overlay_rgb[:, :, c] + 
                               (1 - overlay_alpha) * roi[:, :, c])
            
            background[y:y+height, x:x+width] = roi
        else:
            # No alpha channel, just copy
            background[y:y+height, x:x+width] = overlay_resized
        
        return background
    
    def apply_filter(self, image, filter_name):
        """
        Apply specific filter by name - each filter has custom sizing
        
        Args:
            image: Input image
            filter_name: Name of the filter (e.g., 'sunglasses', 'chef_hat', 'crown')
        
        Returns:
            Image with filter applied
        """
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        # Check if filter exists
        if filter_name not in self.overlays:
            print(f"⚠️ Filter '{filter_name}' not found in loaded overlays")
            return image
        
        # === CUSTOM SETTINGS FOR EACH FILTER ===
        
        # SUNGLASSES - Centered on face
        if filter_name == 'sunglasses':
            # Simple center positioning on face
            filter_width = int(w * 0.9)  # 90% of face width (reduced from 1.2)
            filter_height = int(filter_width * 1.15)  # Height ratio (increased from 0.4)
            
            # Center horizontally on face
            filter_x = x + (w - filter_width) // 2
            
            # Position higher on face (move up from center)
            filter_y = y + (h - filter_height) // 2 - int(h * 0.1)  # Move up by 10% of face height
            
            result = self.overlay_image(result, self.overlays[filter_name], 
                                      filter_x, filter_y, filter_width, filter_height)
            return result
        
        # CHEF HAT
        elif filter_name == 'chef_hat':
            filter_width = int(w * 1.65)
            filter_height = int(filter_width * 0.75)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.85)
        
        # CROWN
        elif filter_name == 'crown':
            filter_width = int(w * 1.3)
            filter_height = int(filter_width * 0.75)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.85)
        
        # GRADUATION HAT
        elif filter_name == 'graduation_hat':
            filter_width = int(w * 1.8)
            filter_height = int(filter_width * 0.75)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.6)  # Lower position
        
        # PARTY HAT
        elif filter_name == 'party_hat':
            filter_width = int(w * 1.3)
            filter_height = int(filter_width * 1)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.95)  # Higher position
        
        # DOG FILTER
        elif filter_name == 'dog':
            filter_width = int(w * 1.5)
            filter_height = int(filter_width * 1)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.2)
        
        # FOX FILTER
        elif filter_name == 'fox':
            filter_width = int(w * 1.0)  # Smaller than other face overlays
            filter_height = int(filter_width * 1)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.2)
        
        # PUMPKIN FILTER
        elif filter_name == 'pumpkin':
            filter_width = int(w * 1.5)
            filter_height = int(filter_width * 1)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.2)
        
        # HEADPHONE FILTER
        elif filter_name == 'headphone':
            filter_width = int(w * 2.0)  # Wider (increased from 1.8)
            filter_height = int(filter_width * 0.75)  # Reduced height (from 1 to 0.6)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.4)  # Moved up a bit (from 0.2 to 0.4)
        
        # HEART FILTER - Above the head
        elif filter_name == 'heart':
            filter_width = int(w * 1.8)  # Wider (increased from 1.5)
            filter_height = int(filter_width * 1)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.95)  # Higher (increased from 0.85)
        
        # DEFAULT for any other filters
        else:
            filter_width = int(w * 1.5)
            filter_height = int(filter_width * 1)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.2)
        
        result = self.overlay_image(result, self.overlays[filter_name], 
                                  filter_x, filter_y, filter_width, filter_height)
        return result
    
    def get_available_filters(self):
        """Return list of all available filter names"""
        return list(self.overlays.keys())
    
    def apply_sunglasses(self, image):
        """Apply sunglasses overlay from custom image"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        # Check if custom sunglasses overlay exists
        if 'sunglasses' in self.overlays:
            eyes = self.detect_eyes(image, face)
            
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                left_eye_x = eyes[0][0] + eyes[0][2] // 2
                left_eye_y = eyes[0][1] + eyes[0][3] // 2
                right_eye_x = eyes[1][0] + eyes[1][2] // 2
                right_eye_y = eyes[1][1] + eyes[1][3] // 2
                
                eye_distance = int(np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2))
                glasses_width = int(eye_distance * 0.5)  # VERY small sunglasses - just 50% of eye distance
                glasses_height = int(glasses_width * 0.5)  # Reduced height ratio too
                
                center_x = (left_eye_x + right_eye_x) // 2
                center_y = (left_eye_y + right_eye_y) // 2
                
                # Position sunglasses much higher - centered on eyes, not nose
                glasses_x = center_x - glasses_width // 2
                glasses_y = center_y - int(glasses_height * 0.55)  # Position so center is at eye level
                
                result = self.overlay_image(result, self.overlays['sunglasses'], 
                                          glasses_x, glasses_y, glasses_width, glasses_height)
            else:
                # Fallback positioning - much higher on face
                glasses_width = int(w * 0.2)  # VERY small - just 20% of face width
                glasses_height = int(glasses_width * 0.5)  # Reduced height ratio
                glasses_x = x + (w - glasses_width) // 2
                glasses_y = y + h // 5  # Moved much higher (1/5 down instead of 1/3)
                
                result = self.overlay_image(result, self.overlays['sunglasses'], 
                                          glasses_x, glasses_y, glasses_width, glasses_height)
        else:
            # Fallback to programmatic drawing
            from ar_filters_simple import ARFiltersSimple
            simple_filters = ARFiltersSimple()
            result = simple_filters.apply_sunglasses(image)
        
        return result
    
    def apply_hat(self, image):
        """Apply hat overlay from custom image"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        if 'hat' in self.overlays:
            # Make hat wider and adjust height based on hat aspect ratio
            hat_width = int(w * 1.8)  # Increased from 1.5 to 1.8 for better coverage
            hat_height = int(hat_width * 0.75)  # Adjusted for better proportions
            hat_x = x + w // 2 - hat_width // 2
            hat_y = y - int(hat_height * 0.85)  # Increased from 0.7 to 0.85 to sit higher on head
            
            result = self.overlay_image(result, self.overlays['hat'], 
                                      hat_x, hat_y, hat_width, hat_height)
        else:
            from ar_filters_simple import ARFiltersSimple
            simple_filters = ARFiltersSimple()
            result = simple_filters.apply_hat(image)
        
        return result
    
    def apply_chef_hat(self, image):
        """Apply chef hat overlay from custom image"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        if 'chef_hat' in self.overlays:
            hat_width = int(w * 1.8)
            hat_height = int(hat_width * 0.75)
            hat_x = x + w // 2 - hat_width // 2
            hat_y = y - int(hat_height * 0.85)
            
            result = self.overlay_image(result, self.overlays['chef_hat'], 
                                      hat_x, hat_y, hat_width, hat_height)
        else:
            # Fallback to generic hat
            return self.apply_hat(image)
        
        return result
    
    def apply_graduation_hat(self, image):
        """Apply graduation hat overlay from custom image"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        if 'graduation_hat' in self.overlays:
            hat_width = int(w * 1.8)
            hat_height = int(hat_width * 0.75)  # Keep original height
            hat_x = x + w // 2 - hat_width // 2
            hat_y = y - int(hat_height * 0.6)  # Reduced from 0.85 to 0.6 to sit lower on head
            
            result = self.overlay_image(result, self.overlays['graduation_hat'], 
                                      hat_x, hat_y, hat_width, hat_height)
        else:
            # Fallback to generic hat
            return self.apply_hat(image)
        
        return result
    
    def apply_party_hat(self, image):
        """Apply party hat overlay from custom image"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        if 'party_hat' in self.overlays:
            hat_width = int(w * 1.3)
            hat_height = int(hat_width * 1)
            hat_x = x + w // 2 - hat_width // 2
            hat_y = y - int(hat_height * 0.95)  # Increased from 0.85 to 0.95 to sit higher
            
            result = self.overlay_image(result, self.overlays['party_hat'], 
                                      hat_x, hat_y, hat_width, hat_height)
        else:
            # Fallback to generic hat
            return self.apply_hat(image)
        
        return result
    
    def apply_dog_filter(self, image):
        """Apply dog filter from custom image (dog.png)"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        # Check if single dog.png exists (preferred method)
        if 'dog' in self.overlays:
            # Use the full face overlay
            filter_width = int(w * 1.5)
            filter_height = int(filter_width * 1)
            filter_x = x + w // 2 - filter_width // 2
            filter_y = y - int(filter_height * 0.2)
            
            result = self.overlay_image(result, self.overlays['dog'], 
                                      filter_x, filter_y, filter_width, filter_height)
            return result
        
        # Fallback: Try separate dog_ears and dog_nose (legacy method)
        if 'dog_ears' in self.overlays:
            ears_width = int(w * 1.8)
            ears_height = int(ears_width * 0.6)
            ears_x = x + w // 2 - ears_width // 2
            ears_y = y - ears_height // 3
            
            result = self.overlay_image(result, self.overlays['dog_ears'], 
                                      ears_x, ears_y, ears_width, ears_height)
        
        if 'dog_nose' in self.overlays:
            nose_size = w // 4
            nose_x = x + w // 2 - nose_size // 2
            nose_y = y + int(h * 0.65)
            
            result = self.overlay_image(result, self.overlays['dog_nose'], 
                                      nose_x, nose_y, nose_size, nose_size)
        
        # Final fallback to programmatic drawing
        if 'dog' not in self.overlays and 'dog_ears' not in self.overlays and 'dog_nose' not in self.overlays:
            from ar_filters_simple import ARFiltersSimple
            simple_filters = ARFiltersSimple()
            result = simple_filters.apply_dog_filter(image)
        
        return result
    
    def apply_crown(self, image):
        """Apply crown overlay from custom image"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        if 'crown' in self.overlays:
            crown_width = int(w * 1.2)
            crown_height = int(crown_width * 0.8)
            crown_x = x + w // 2 - crown_width // 2
            crown_y = y - int(crown_height * 0.6)
            
            result = self.overlay_image(result, self.overlays['crown'], 
                                      crown_x, crown_y, crown_width, crown_height)
        else:
            from ar_filters_simple import ARFiltersSimple
            simple_filters = ARFiltersSimple()
            result = simple_filters.apply_crown(image)
        
        return result
    
    def apply_heart_eyes(self, image):
        """Apply heart eyes overlay from custom image"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        if 'hearts' in self.overlays:
            eyes = self.detect_eyes(image, face)
            heart_size = w // 6
            
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_x = ex + ew // 2 - heart_size // 2
                    eye_y = ey + eh // 2 - heart_size // 2
                    
                    result = self.overlay_image(result, self.overlays['hearts'], 
                                              eye_x, eye_y, heart_size, heart_size)
            else:
                # Fallback positioning
                for offset_x in [w // 3, 2 * w // 3]:
                    eye_x = x + offset_x - heart_size // 2
                    eye_y = y + h // 3 - heart_size // 2
                    
                    result = self.overlay_image(result, self.overlays['hearts'], 
                                              eye_x, eye_y, heart_size, heart_size)
        else:
            from ar_filters_simple import ARFiltersSimple
            simple_filters = ARFiltersSimple()
            result = simple_filters.apply_heart_eyes(image)
        
        return result
    
    def apply_sparkles(self, image):
        """Apply sparkles overlay from custom image"""
        face = self.detect_face(image)
        if face is None:
            return image
        
        result = image.copy()
        x, y, w, h = face
        
        if 'sparkle' in self.overlays:
            sparkle_size = w // 8
            sparkle_positions = [
                (x - sparkle_size, y),
                (x + w // 2 - sparkle_size // 2, y - sparkle_size),
                (x + w, y),
                (x - sparkle_size, y + h // 2),
                (x + w, y + h // 2),
                (x, y + h),
                (x + w - sparkle_size, y + h),
            ]
            
            for (sx, sy) in sparkle_positions:
                result = self.overlay_image(result, self.overlays['sparkle'], 
                                          sx, sy, sparkle_size, sparkle_size)
        else:
            from ar_filters_simple import ARFiltersSimple
            simple_filters = ARFiltersSimple()
            result = simple_filters.apply_sparkles(image)
        
        return result
    
    def close(self):
        """Clean up resources"""
        pass
