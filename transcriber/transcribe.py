import whisper_timestamped as whisper
import torch
import json

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    filepath = "../demo/_YHDxyO9W8Q.mp4"

    audio = whisper.load_audio(filepath)
    model = whisper.load_model("tiny", device=device)
    result = whisper.transcribe(model, audio, language="en")
    with open("../demo/transcript.json", 'w') as f:
        json.dump(result, f)
