from pitch import pitch_apply
from speed import speed_apply
from age_gender import age_gender_apply
import librosa
import json
import bisect

SPEAKER_RATE_BINS = ["very slowly", "slowly", "slightly slowly", "moderate speed", "slightly fast", "fast", "very fast"]
UTTERANCE_LEVEL_STD = ["very monotone", "monotone", "slightly expressive and animated", "expressive and animated", "very expressive and animated"]
SPEAKER_LEVEL_PITCH_BINS = ["very low-pitch", "low-pitch", "slightly low-pitch", "moderate pitch", "slightly high-pitch", "high-pitch", "very high-pitch"]
with open("bin.json") as json_file:
    text_bins_dict = json.load(json_file)

audiopath = "YOUR_AUDIO_PATH"
waveform, _ = librosa.load(audiopath, sr=16000)
age, gender = age_gender_apply(waveform)
pitch_mean, pitch_std = pitch_apply(waveform)
if gender == "male":
    index = bisect.bisect_right(text_bins_dict["pitch_bins_male"], pitch_mean) - 1
    pitch = SPEAKER_LEVEL_PITCH_BINS[index]
else:
    index = bisect.bisect_right(text_bins_dict["pitch_bins_female"], pitch_mean) - 1
    pitch = SPEAKER_LEVEL_PITCH_BINS[index]

index = bisect.bisect_right(text_bins_dict["speech_monotony"], pitch_std) - 1
monotony = UTTERANCE_LEVEL_STD[index]
speech_duration = speed_apply(waveform)

index = bisect.bisect_right(text_bins_dict["speaking_rate"], speech_duration) - 1
speed = SPEAKER_RATE_BINS[index]

print(pitch, monotony, speed, age, gender)
