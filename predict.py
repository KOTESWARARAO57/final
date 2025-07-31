import cv2
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
from extract_features import extract_emotion_from_video, extract_text_from_audio
from utils import translate_text, speak_text
import pickle

def record_audio(filename="input.wav", duration=5):
    print("🎙️ Speak now (in Telugu)...")
    fs = 44100
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)

def capture_video(filename="capture.avi", duration=5):
    print("📷 Capturing video...")
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    for _ in range(duration * 20):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video("test.avi")
    record_audio("test.wav")

    emotion = extract_emotion_from_video("test.avi")
    print("Detected Emotion:", emotion)

    text_te = extract_text_from_audio("test.wav")
    print("Telugu Speech:", text_te)

    text_en = translate_text(text_te, src="te", target="en")
    print("Translated to English:", text_en)

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    pred = model.predict([[hash(emotion) % 100]])[0]
    print("🧠 Predicted Disease:", pred)

    pred_te = translate_text(pred, src="en", target="te")
    speak_text(pred_te)
