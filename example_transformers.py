from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

from tqdm import tqdm

import librosa

# Set to True to skip timestamp
no_timestamps = False

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

data, samplerate = librosa.load("tests/jfk.flac", sr=16000)
# load dummy dataset and read soundfiles
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

input_features = processor(data, return_tensors="pt").input_features

# it is not enough to set no_timestamp to False, often the model still predticts it as third token
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=no_timestamps)
if no_timestamps is False:
    model.config.suppress_tokens.extend(processor.tokenizer.encode("<|notimestamps|>"))

# decoding strategies
# greedy
# sample
# beam
# beam-sample
# beam-group
resps = []
for i in tqdm(range(1)):
    resp = model.generate(
        input_features,
        do_sample=True,
        num_return_sequences=1,
        num_beams=1,
        num_beam_groups=1
    )
    try:
        resps.extend(processor.batch_decode(resp["sequences"], skip_special_tokens = True))
    except (IndexError, KeyError):
        resps.extend(processor.batch_decode(resp, skip_special_tokens = True))

print(resps)
