# hwang258@jhu.edu

import torch
import capspeech.nar.generate as nar
from transformers import AutoTokenizer, set_seed
import soundfile as sf
import time
import os
from huggingface_hub import snapshot_download
import argparse

# You can try these examples for CapTTS
examples_CapTTS = [
    {
        "transcript":"From these genres and from these spaces, you know, and the feelings of what these games can bring.",
        "caption":"An elderly woman, with a low-pitched voice, delivers her speech in a slow, yet expressive and animated manner. Her words flow like a captivating story, each sentence filled with emotion and wisdom, resonating deeply with her audience."
    },
    {
        "transcript":"Use hir or ze as gender neutral pronouns?",
        "caption":"A male speaker delivers his words in a measured pace, exhibiting a high-pitched, happy, and animated tone in a clean environment."
    },
    {
        "transcript":"in germany they generally hock the kaiser",
        "caption":"Her voice, a combination of feminine allure and intellectual brilliance, resonates with a sense of calm and elegance, making her every word a testament to her cool sophistication."
    },
    {
        "transcript":"i want to see what he was when he was bright and young before the world had hardened him",
        "caption":"A mature male voice, rough and husky, ideal for public speaking engagements."
    }
]

# You can try these examples for AgentTTS
examples_AgentTTS = [
    {
        "transcript":"If only I had pursued my passion for dance earlier, I could have become a professional dancer.",
        "caption":"A voice tinged with regret, conveying a sense of longing for what could have been."
    },
    {
        "transcript":"Please, step lightly on the path ahead, respecting nature's delicate balance as we make our way forward.",
        "caption":"Quiet, submissive intonation reflecting a sense of yielding and compliance."
    },
    {
        "transcript":"The intricate patterns and vibrant colors of each quilt showcase the love and dedication poured into every stitch.",
        "caption":"Sincere and soft-spoken voice filled with kindness and compassion."
    },
    {
        "transcript":"Just reboot the robot and step aside, clearly, this is beyond your expertise.",
        "caption":"Cool, dismissive tone conveying disdain."
    }
]

# You can try these examples for EmoCapTTS
examples_EmoCapTTS = [
    {
        "transcript":"Why does your car smell like a dead RAT? It's absolutely vile.",
        "caption":"A middle-aged woman speaks in a low, monotone voice, her words dripping with disgust and annoyance."
    },
    {
        "transcript":"Dark smoke billowed from the engine as the car refused to start.",
        "caption":"	A middle-aged woman speaks in a low, slow, monotone tone, her words dripping with a palpable, simmering frustration."
    },
    {
        "transcript":"The cat, usually aloof, suddenly curled up in his lap, purring contentedly.",
        "caption":"A man, speaking with a slightly high-pitched, slow, and slightly expressive tone, conveying surprise with a gentle, bemused warmth."
    },
    {
        "transcript":"The distant hum of traffic passed unnoticed in the background.",
        "caption":"She speaks in a calm, measured tone, conveying a sense of quiet, unassuming presence."
    }
]


# You can try these examples for AccCapTTS
examples_AccCapTTS = [
    {
        "transcript":"This is a sector in overall deficit and urgent action is required.",
        "caption":"An Indian-accented professional woman's voice for client and public interaction."
    },
    {
        "transcript":"When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
        "caption":"A mature, deep, and rough Scottish-accented female voice, with a hint of weakness, ideal for client and public interaction in customer service or community relations."
    },
    {
        "transcript":"It will decide today whether he should be struck off the register.",
        "caption":"A young girl's English-accented voice, suitable for customer service and public engagement roles."
    },
    {
        "transcript":"If we live forever, can we really be said to live?",
        "caption":"Bright teenage girl's voice, with an American accent, ideal for client and public interaction."
    },
    {
        "transcript":"Subs not used, McGraw.",
        "caption":"A mature, Australian person voice, deep and rough."
    }
]

def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if not args.random:
        nar.seed_everything(args.seed)
    model_list = nar.load_model(device, args.task)
    audio_arr = nar.run(model_list, device, args.duration, args.transcript, args.caption)
    sf.write(args.output_path, audio_arr, 24000) # sampling rate fixed in this work
    print(f"Save to : {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=["PT", "CapTTS", "EmoCapTTS", "AccCapTTS", "AgentTTS"], help="Task to choose")
    parser.add_argument('--seed', type=int, default=42, help="Fixed seed")
    parser.add_argument("--random", action="store_true", help="Enable a random seed, otherwise use a fixed seed")
    parser.add_argument('--duration', type=float, default=None, help="Set a fixed duration for output audio")
    parser.add_argument('--transcript', type=str, required=True, help="Transcript of audio")
    parser.add_argument('--caption', type=str, required=True, help="style caption of audio")
    parser.add_argument('--output_path', type=str, required=True, help="Output path to save audio file")
    args = parser.parse_args()
    main(args)
