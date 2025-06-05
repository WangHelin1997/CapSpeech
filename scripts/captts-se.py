# hwang258@jhu.edu

import torch
from capspeech.ar.parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf
import time
import os
from huggingface_hub import snapshot_download
import argparse

# You can try these examples
examples = [
    {
        "transcript":"<dog> at this moment miss brandon entered with her brilliant cousin rachel the blonde and the dark it was a dazzling contrast <I_start> <I_end>",
        "caption":"A young woman speaks at a moderate pace, her voice carrying a hint of monotone. Remarkably, she maintains a high pitch, giving her speech an air of focused determination."
    },
    {
        "transcript":"<coughing> it did not beckon or indeed move at all <I_start> <I_end> it was as still as the hand of death",
        "caption":"A middle-aged man delivers his words in a monotone manner, maintaining a steady, moderate pace. His slightly elevated pitch, while not overly dramatic, imparts a subtle sense of gravity to his speech."
    },
    {
        "transcript":"<cat> <I_start> <I_end> he stood still in deference to their calls and parried their banter with easy words",
        "caption":"The speech emanates from a middle-aged man, his voice resonating with a slight, low-pitch timbre. His delivery is steady, devoid of emotive inflections, yet not entirely monotonous. The moderate pace of his speech lends an air of thoughtful consideration to his words."
    },
    {
        "transcript":"<clapping> i know said margaret bolton with a half anxious smile <B_start> the chafes against all the ways of friends <B_end> but what will thee do",
        "caption":"An old female speaker, her voice subtly animated, adopts a slightly low pitch. Her speech pace is moderate."
    },
    {
        "transcript":"<keyboard_typing> <B_start> at the inception of plural marriage among the latter day saints <B_end> there was no law national or state against its practise",
        "caption":"A middle-aged man, his voice unwavering and steady, delivers his speech in a monotone manner. The moderate pitch, neither too high nor too low, adds a touch of gravitas to his words. The slight fast pace hints at a sense of urgency or importance, yet his tone remains consistently measured and controlled."
    }
]

def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print("Downloading model from Huggingface...")
    local_dir = snapshot_download(
        repo_id="OpenSound/CapSpeech-models"
    )
    model_path = os.path.join(local_dir, "ar_CapTTS-SE")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Start Generation...")
    start_time = time.time()
    input_ids = tokenizer(args.caption, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(args.transcript, return_tensors="pt").input_ids.to(device)
    if not args.random:
        set_seed(args.seed)
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, guidance_scale=args.guidance_scale)
    audio_arr = generation.cpu().numpy().squeeze()
    end_time = time.time()
    audio_len = audio_arr.shape[-1] / model.config.sampling_rate
    rtf = (end_time-start_time)/audio_len
    print(f"RTF: {rtf:.4f}")
    sf.write(args.output_path, audio_arr, model.config.sampling_rate)
    print(f"Save to : {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="Fixed seed")
    parser.add_argument("--random", action="store_true", help="Enable a random seed, otherwise use a fixed seed")
    parser.add_argument('--transcript', type=str, required=True, help="Transcript of audio")
    parser.add_argument('--caption', type=str, required=True, help="style caption of audio")
    parser.add_argument('--output_path', type=str, required=True, help="Output path to save audio file")
    parser.add_argument('--guidance_scale', type=float, default=1.5, help="CFG guidance scale")
    args = parser.parse_args()
    main(args)
