import cv2
import numpy as np
from PIL import Image, ImageEnhance

class ImageEditor:
    
    @staticmethod
    def resize_image(image, width=None, height=None, scale=None):
        h, w = image.shape[:2]
        
        if scale:
            new_w = int(w * scale)
            new_h = int(h * scale)
        elif width and height:
            new_w, new_h = width, height
        elif width:
            new_w = width
            new_h = int(h * (width / w))
        elif height:
            new_h = height
            new_w = int(w * (height / h))
        else:
            return image
            
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def rotate_image(image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(image, matrix, (new_w, new_h), 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=(255, 255, 255))
    
    @staticmethod
    def crop_image(image, x, y, width, height):
        h, w = image.shape[:2]
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        width = min(width, w - x)
        height = min(height, h - y)
        
        return image[y:y+height, x:x+width]
    
    @staticmethod
    def adjust_brightness(image, factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_contrast(image, factor):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def apply_filter(image, filter_name):
        if filter_name == "Grayscale":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif filter_name == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
            return cv2.transform(image, kernel)
        
        elif filter_name == "Blur":
            return cv2.GaussianBlur(image, (15, 15), 0)
        
        elif filter_name == "Sharpen":
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        
        elif filter_name == "Edge Detection":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        elif filter_name == "Vintage":
            sepia = ImageEditor.apply_filter(image, "Sepia")
            hsv = cv2.cvtColor(sepia, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * 0.6
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        elif filter_name == "Cool":
            result = image.copy().astype(np.float32)
            result[:, :, 0] = np.clip(result[:, :, 0] * 1.2, 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * 0.9, 0, 255)
            return result.astype(np.uint8)
        
        elif filter_name == "Warm":
            result = image.copy().astype(np.float32)
            result[:, :, 2] = np.clip(result[:, :, 2] * 1.2, 0, 255)
            result[:, :, 0] = np.clip(result[:, :, 0] * 0.9, 0, 255)
            return result.astype(np.uint8)
        
        return image
    
    @staticmethod
    def flip_image(image, direction):
        if direction == "horizontal":
            return cv2.flip(image, 1)
        elif direction == "vertical":
            return cv2.flip(image, 0)
        return image
