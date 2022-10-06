import re
import time

import whisper
from whisper.audio import StreamAudio
# from whisper.tokenizer import get_tokenizer

model = whisper.load_model("small")


def _align_text(old_text, new_text):
    new_text = re.sub(r'[^\w\s]', '', new_text).lower()
    if old_text is None:
        # first iteration
        return new_text

    # beginning
    for i in range(len(old_text)):
        if new_text.startswith(old_text[i:]):
            break
    
    # end
    # example of problem to solve
    # old_text: and so my fellow americans ask not what youre coming
    # new_text: ask not what your country can do for you
    # current_output: and so my fellow americans ask not what youre cominask not what your country can do for you
    # expected output: and so my fellow americans ask not what your country can do for you

    return old_text[:i] + new_text


# load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("tests/jfk.flac")
filepath = "tests/jfk.flac"
lang = None
text = None
with StreamAudio(filepath, max_duration=5) as audio_stream:
    audio_generator = audio_stream.generator()
    for i, chunk in enumerate(audio_generator):
        t = time.time()
        audio = whisper.pad_or_trim(chunk, trim_start=True)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        options = whisper.DecodingOptions(fp16=False)

        # detect the spoken language
        if lang is None:
            _, probs = model.detect_language(mel)
            lang = max(probs, key=probs.get)
            # tokenizer = get_tokenizer(model.is_multilingual, language=lang, task=options.task)
            print(f"Detected language: {lang}")

        # decode the audio
        result = whisper.decode(model, mel, options)
        text = _align_text(text, result.text)

        # print the recognized text
        print(text, end="\r")

        # emulate realtime streaming
        dt = time.time() - t
        time.sleep(max(0, 1 - dt)) 

    print(text)
