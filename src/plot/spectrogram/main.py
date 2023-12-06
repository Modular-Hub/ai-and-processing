import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_file = "Old Soul.mp3"
y, sr = librosa.load(audio_file)

spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
spectrogram_db = librosa.power_to_db(spectrogram)

plt.figure(figsize=(10, 5))
librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')

plt.savefig('spectrogram.png')
plt.show()
