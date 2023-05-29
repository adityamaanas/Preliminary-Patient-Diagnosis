#import library
import speech_recognition as sr
import time

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable

with sr.Microphone() as source:
    print("Please wait. Calibrating microphone...")   
    # listen and create the ambient noise energy level   
    r.adjust_for_ambient_noise(source, duration=5)  
    r.dynamic_energy_threshold = True
    print("Talk")
    audio_text = r.listen(source,timeout=5, phrase_time_limit= 15)
    print("Time over, thanks")

# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:
        # using google speech recognition
        print("Text: "+r.recognize_google(audio_text))
    except sr.UnknownValueError:
        print("I didn't understand what you said.")
    except sr.RequestError as e:
         print("Could not request results from Google Speech Recognition service; {0}".format(e))