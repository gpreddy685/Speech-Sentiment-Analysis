import transformers
import librosa, torch
import streamlit as st
from audiorecorder import audiorecorder
from transformers import AutoModelForAudioClassification

a_model = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Categorical-Attributes", trust_remote_code=True)

def record_audio():
    st.title("Audio Recorder")
    audio = audiorecorder("Click to record", "Click to stop recording")
    audio_file_path = None
    if len(audio) > 0:
      st.audio(audio.export().read())
      audio_file_path = "Audio" + ".wav"
      audio.export(audio_file_path, format="wav")
    st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
    return audio_file_path

def predict_tone(audio_file_path, a_model):
    if not audio_file_path:
        return None

    mean = a_model.config.mean
    std = a_model.config.std

    raw_wav, _ = librosa.load(audio_file_path, sr=a_model.config.sampling_rate)
    norm_wav = (raw_wav - mean) / (std+0.000001)
    mask = torch.ones(1, len(norm_wav))
    wavs = torch.tensor(norm_wav).unsqueeze(0)
    with torch.no_grad():
        pred = a_model(wavs, mask)
    tone_index = torch.argmax(pred)
    tone = a_model.config.id2label[tone_index.item()]

    return tone

st.info('Record your voice 🌝', icon=None)
audio_file_name = record_audio()
if audio_file_name:
  st.button("Analyse the Sentiment")
  tone = predict_tone(audio_file_name, a_model)
  if tone is not None:
        st.info(f"Detected tone: {tone}", icon=None) 