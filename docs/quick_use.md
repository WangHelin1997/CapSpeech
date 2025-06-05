# Quick Use

We release our 🚀 best models for CapTTS, CapTTS-SE, AccCapTTS, EmoCapTTS, and AgentTTS. This page provides a quick-start guide for using them. 

⏳ Note that the first run may take some time as it needs to download the pretrained checkpoints.

## Table of Content

- [Install](#install): how to install on linux
- [CapTTS](#captts): how to use our CapTTS model
- [CapTTSSE](#capttsse): how to use our CapTTS-SE model
- [AccCapTTS](#acccaptts): how to use our AccCapTTS model
- [EmoCapTTS](#emocaptts): how to use our EmoCapTTS model
- [AgentTTS](#agenttts): how to use our AgentTTS model

## Install

For Linux developers and researchers, run:

```bash
conda create -n capspeech python=3.10
conda activate capspeech
pip install git+https://github.com/WangHelin1997/CapSpeech.git
```

## CapTTS

Please see [CapTTS_demo.ipynb](../scripts/CapTTS_demo.ipynb) for an example usage.

You can also run this file:
```bash
python scripts/captts.py \
    --task "CapTTS" \
    --transcript "Use hir or ze as gender neutral pronouns?" \
    --caption "A male speaker delivers his words in a measured pace, exhibiting a high-pitched, happy, and animated tone in a clean environment." \
    --output_path "./demo/test_captts.wav"
```
Feel free to adjust `--seed` to generate a different reproducible sample. To produce a different sample on each run, use `--random`. You can also modify `--duration` to suit your preference.

## CapTTSSE

⚠️ The transcripts in CapTTS-SE models support these two modes:

1. Insertion: starts with a sound event tag, and `<I_start> <I_end>` represents the insertion position, e.g.
```
<dog> at this moment miss brandon entered with her brilliant cousin rachel the blonde and the dark it was a dazzling contrast <I_start> <I_end>
```
2. Background: starts with a sound event tag, and `<B_start>`, `<B_end>` represent the start and end of the background sound, e.g.
```
<clapping> i know said margaret bolton with a half anxious smile <B_start> the chafes against all the ways of friends <B_end> but what will thee do
```

📝 Recommended sound events:
```
<dog>
<cat>
<coughing>
<laughing>
<clapping>
<footsteps>
<door_wood_knock>
<clock_alarm>
<keyboard_typing>
<can_opening>
```

Please see [CapTTS-SE_demo.ipynb](../scripts/CapTTS-SE_demo.ipynb) for an example usage.

You can also run this file:
```bash
python scripts/captts-se.py \
    --transcript "<dog> at this moment miss brandon entered with her brilliant cousin rachel the blonde and the dark it was a dazzling contrast <I_start> <I_end>" \
    --caption "A young woman speaks at a moderate pace, her voice carrying a hint of monotone. Remarkably, she maintains a high pitch, giving her speech an air of focused determination." \
    --output_path "./demo/test_capttsse.wav"
```
Feel free to adjust `--seed` to generate a different reproducible sample. To produce a different sample on each run, use `--random`.

## AccCapTTS

Please see [AccCapTTS_demo.ipynb](../scripts/AccCapTTS_demo.ipynb) for an example usage.

You can also run this file:
```bash
python scripts/captts.py \
    --task "AccCapTTS" \
    --transcript "This is a sector in overall deficit and urgent action is required." \
    --caption "An Indian-accented professional woman's voice for client and public interaction." \
    --output_path "./demo/test_acccaptts.wav"
```
Feel free to adjust `--seed` to generate a different reproducible sample. To produce a different sample on each run, use `--random`. You can also modify `--duration` to suit your preference.

## EmoCapTTS

Please see [EmoCapTTS_demo.ipynb](../scripts/EmoCapTTS_demo.ipynb) for an example usage.

You can also run this file:
```bash
python scripts/captts.py \
    --task "EmoCapTTS" \
    --transcript "Why does your car smell like a dead RAT? It's absolutely vile." \
    --caption "A middle-aged woman speaks in a low, monotone voice, her words dripping with disgust and annoyance." \
    --output_path "./demo/test_emocaptts.wav"
```
Feel free to adjust `--seed` to generate a different reproducible sample. To produce a different sample on each run, use `--random`. You can also modify `--duration` to suit your preference.


## AgentTTS

⚠️ The AgentTTS model is under safety verification and will be made publicly available soon.

Please see [AgentTTS_demo.ipynb](../scripts/AgentTTS_demo.ipynb) for an example usage.

You can also run this file:
```bash
python scripts/captts.py \
    --task "AgentTTS" \
    --transcript "The intricate patterns and vibrant colors of each quilt showcase the love and dedication poured into every stitch." \
    --caption "Sincere and soft-spoken voice filled with kindness and compassion." \
    --output_path "./demo/test_agenttts.wav"
```
Feel free to adjust `--seed` to generate a different reproducible sample. To produce a different sample on each run, use `--random`. You can also modify `--duration` to suit your preference.