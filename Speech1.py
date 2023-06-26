import pickle
import librosa
import soundfile
import numpy as np
from sklearn.neural_network import MLPClassifier

# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Define the MLP Classifier model
model = MLPClassifier(
    alpha=0.01,
    batch_size=256,
    epsilon=1e-08,
    hidden_layer_sizes=(300,),
    learning_rate='adaptive',
    max_iter=500
)

# Load the trained model
def load_ser_model():
    with open('emotion.sav', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the SER model
model = load_ser_model()

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Generate a response based on the predicted emotion
def generate_response(predicted_emotion):
    # Define your response generation logic here
    # You can use the predicted_emotion to generate an appropriate response
    response = "The predicted emotion is: " + predicted_emotion
    return response

# Predict the emotion given an audio file
def predict_emotion(audio_file):
    feature = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    predicted_emotion = model.predict([feature])[0]
    print(predicted_emotion)
    return predicted_emotion