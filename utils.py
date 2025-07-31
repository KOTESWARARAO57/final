from deep_translator import GoogleTranslator
from gtts import gTTS
import os

def translate_text(text, src='te', target='en'):
    return GoogleTranslator(source=src, target=target).translate(text)

def speak_text(text, lang='te'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("start output.mp3")
