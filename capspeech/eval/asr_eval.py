from jiwer import wer as calculate_wer
from jiwer import cer as calculate_cer
from whisper.normalizers import EnglishTextNormalizer
import whisper
import torch

normalizer = EnglishTextNormalizer()
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large-v3-turbo", device=device)

def asr(wav_path):
    result = whisper_model.transcribe(wav_path)
    pred = result['text'].strip()
    pred = normalizer(pred)
    return pred

if __name__ == '__main__':
    gt_text="Hey, how are you doing today? I like it."
    wav_path="your-audio"
    gt_text = normalizer(gt_text.strip())
    pred_asr = asr(wav_path)
    wer = round(calculate_wer(gt_text, pred_asr), 3)
    cer = round(calculate_cer(gt_text, pred_asr), 3)
    print(wer, cer)