import pygame
import os
from pathlib import Path

class MusicController:
    
    def __init__(self):
        pygame.mixer.init()
        self.current_music = None
        self.is_playing = False
        self.volume = 0.5
        
    def load_music(self, music_file):
        try:
            if os.path.exists(music_file):
                pygame.mixer.music.load(music_file)
                self.current_music = music_file
                return True
            else:
                print(f"Music file not found: {music_file}")
                return False
        except Exception as e:
            print(f"Error loading music: {e}")
            return False
    
    def play(self, loops=-1):
        try:
            if self.current_music:
                pygame.mixer.music.play(loops=loops)
                self.is_playing = True
                pygame.mixer.music.set_volume(self.volume)
        except Exception as e:
            print(f"Error playing music: {e}")
    
    def stop(self):
        try:
            pygame.mixer.music.stop()
            self.is_playing = False
        except Exception as e:
            print(f"Error stopping music: {e}")
    
    def pause(self):
        try:
            pygame.mixer.music.pause()
            self.is_playing = False
        except Exception as e:
            print(f"Error pausing music: {e}")
    
    def unpause(self):
        try:
            pygame.mixer.music.unpause()
            self.is_playing = True
        except Exception as e:
            print(f"Error unpausing music: {e}")
    
    def set_volume(self, volume):
        try:
            self.volume = max(0, min(100, volume)) / 100.0
            pygame.mixer.music.set_volume(self.volume)
        except Exception as e:
            print(f"Error setting volume: {e}")
    
    def get_volume(self):
        return int(self.volume * 100)
    
    def is_music_playing(self):
        return pygame.mixer.music.get_busy()
    
    def fadeout(self, milliseconds=1000):
        try:
            pygame.mixer.music.fadeout(milliseconds)
            self.is_playing = False
        except Exception as e:
            print(f"Error fading out music: {e}")
    
    def get_music_info(self):
        if self.current_music:
            return {
                'file': os.path.basename(self.current_music),
                'path': self.current_music,
                'volume': self.get_volume(),
                'is_playing': self.is_music_playing()
            }
        return None
    
    def cleanup(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception as e:
            print(f"Error cleaning up: {e}")

FILTER_MUSIC_MAP = {
    "Sunglasses": "cool_vibes.mp3",
    "Hat": "jazz.mp3",
    "Dog Filter": "playful.mp3",
    "Crown": "royal.mp3",
    "Heart Eyes": "romantic.mp3",
    "Sparkles": "magical.mp3"
}

def get_music_for_filter(filter_name, music_dir="music"):
    if filter_name in FILTER_MUSIC_MAP:
        music_file = os.path.join(music_dir, FILTER_MUSIC_MAP[filter_name])
        if os.path.exists(music_file):
            return music_file
    
    default_music = os.path.join(music_dir, "default.mp3")
    if os.path.exists(default_music):
        return default_music
    
    return None
