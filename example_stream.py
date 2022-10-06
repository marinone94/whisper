import whisper
from whisper.audio import StreamAudio

model = whisper.load_model("small")

# load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("tests/jfk.flac")
filepath = "tests/jfk.flac"
lang = None
with StreamAudio(filepath) as audio_stream:
    audio_generator = audio_stream.generator()
    for chunk in audio_generator:
        audio = whisper.pad_or_trim(chunk, trim_start=True)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        if lang is None:
            _, probs = model.detect_language(mel)
            lang = max(probs, key=probs.get)
            print(f"Detected language: {lang}")

        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        # print the recognized text
        text = result.text
        print(text, end="\r")
    
    print(text)
