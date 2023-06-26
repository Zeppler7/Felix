
import speech as s
import pickle
import Felix
import speech_recognition as sr
wav = None
def get_command():
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        rec.pause_threshold = 1

        audio = rec.listen(source,timeout=3,phrase_time_limit=5)
        global wav
        wav = audio.get_wav_data()
        
        #Speech1.predict_emotion(wav)
        try:
            print("Recognizing...")
            query = rec.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return "None"
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return "None"

        except Exception as e:
            print(e)
            print("Say that again please...")
            return "None"
        return query
emotion = s.extract_feature(wav, mfcc=True, chroma=True, mel=True)
#model = pickle.load(open('emotion.sav', 'rb'))
#y_pred = model.predict([emotion])
print(emotion)
#print(y_pred)