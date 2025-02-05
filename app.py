import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from huggingface_hub import login
from pydub import AudioSegment
from audiorecorder import audiorecorder
import io, numpy as np
import wave
import os

# ✅ Install required packages at runtime (for Streamlit Cloud)
os.system("pip install torch transformers pydub audiorecorder numpy wave ffmpeg-python huggingface_hub")

st.set_page_config(layout="wide")

# ✅ Authenticate Hugging Face account
HF_TOKEN = "your_huggingface_access_token"  # 🔹 Replace with your token from https://huggingface.co/settings/tokens
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)

# ✅ Set device (use CPU if MPS is unavailable)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ✅ Initialize models with authentication
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", use_auth_token=True)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", use_auth_token=True).to(device)

    # Initialize Llama model for text generation
    pipe = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-1B-Instruct",
        device=0,  # Use CPU/GPU
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=True
    )

    # Initialize TTS model
    tts_model = pipeline("text-to-speech", model="facebook/mms-tts-eng", use_auth_token=True)

    return processor, model, pipe, tts_model

# ✅ Load models
processor, model, pipe, tts_model = load_models()

# ✅ Transcribe audio
def transcribe_audio(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1)
    raw_audio = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0  

    inputs = processor(raw_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# ✅ Generate response using Llama
def generate_response(transcription):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant. Summarize your response in 80-100 words"},
        {"role": "user", "content": transcription},
    ]
    
    outputs = pipe(conversation, max_new_tokens=150)
    return outputs[0]["generated_text"][-1]["content"]

# ✅ Convert text to speech
def text_to_speech(response):
    speech = tts_model(response)
    audio_array = speech["audio"]
    sample_rate = speech["sampling_rate"]

    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_array = (audio_array * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())

    buffer.seek(0)
    return buffer.read()

# ✅ Streamlit Interface
st.title("Voice Bot (English)")

audio = audiorecorder(start_prompt="", stop_prompt="")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")
    st.write("Transcribing the recorded audio...")
    
    audio_bytes = audio.export().read()  
    transcription = transcribe_audio(audio_bytes)

    st.markdown("### User Prompt:")
    st.markdown(f"##### {transcription}")

    st.write("Generating the response...")
    response = generate_response(transcription)

    st.markdown("### AI Assistant:")
    st.markdown(f"##### {response}")

    st.write("Generating speech from response...")
    audio_response = text_to_speech(response)
    st.audio(audio_response, format="audio/wav")
