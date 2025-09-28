from .config import SpeechConfig
from .voice import Voice
import os 
import wave
import requests

def download_voice(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"Downloading {url} ...")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path
    
def load_voice(voice_name="th_f_1"):
    
    model_filename = f"{voice_name}.onnx"
    config_filename = f"{voice_name}.onnx.json"

    local_model_path = f"./voices/{model_filename}"
    local_config_path = f"./voices/{config_filename}"

    if os.path.exists(local_model_path) and os.path.exists(local_config_path):
        return Voice.load(local_model_path, local_config_path)
    else:
        model_url = f"https://huggingface.co/VIZINTZOR/VachanaTTS/resolve/main/voices/{model_filename}"
        config_url = f"https://huggingface.co/VIZINTZOR/VachanaTTS/resolve/main/speaker_config.json"

        model_path = download_voice(model_url, local_model_path)
        config_path = download_voice(config_url, local_config_path)

        return Voice.load(model_path, config_path)

def TTS(
    text,
    voice="th_f_1",
    output="output.wav",
    volume=1.0,
    speed=1.0,
    noise_scale=0.667,
    noise_w_scale=0.8
):

    syn_config = SpeechConfig(
        volume=volume,
        length_scale=(1 / speed), 
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale, 
    )

    voice = load_voice(voice)
    print(voice.synthesize(text, syn_config))
    return voice.synthesize(text, syn_config)
    # with wave.open(output, "wb") as wav_file:
    #     voice.synthesize_wav(text, wav_file, syn_config)