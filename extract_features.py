import cv2
import numpy as np
from deepface import DeepFace
import whisper
import tempfile
import os
from pydub import AudioSegment

def extract_emotion_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions.append(result[0]['emotion'])
        except:
            continue

    cap.release()
    if emotions:
        avg_emotion = max(set([max(e, key=e.get) for e in emotions]), key = [max(e, key=e.get) for e in emotions].count)
        return avg_emotion
    return "neutral"

def extract_text_from_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="te")
    return result["text"]
