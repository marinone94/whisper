import whisper

model = whisper.load_model("large")
result = model.transcribe("tests/03_29_04_55931391216649608242007207894050878714.wav", task="transcribe")
print(result["text"])