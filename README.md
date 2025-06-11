<img src="assets/capspeech_logo.png">
<h3  align="center">🧢 CapSpeech: Enabling Downstream Applications in Style-Captioned Text-to-Speech</h3>

<p align="center">
  📄 <a href="https://arxiv.org/abs/2506.02863"><strong>Paper</strong></a> &nbsp;|&nbsp;
  🌐 <a href="https://wanghelin1997.github.io/CapSpeech-demo/"><strong>Project Page</strong></a> &nbsp;|&nbsp;
  🗂 <a href="https://huggingface.co/datasets/OpenSound/CapSpeech"><strong>Dataset</strong></a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/OpenSound/CapSpeech-models/"><strong>Models</strong></a> &nbsp;|&nbsp;
  🚀 <a href="https://huggingface.co/spaces/OpenSound/CapSpeech-TTS/"><strong>Live Demo</strong></a>
</p>

<p align="center">
  <!-- <img src="https://visitor-badge.laobi.icu/badge?page_id=WangHelin1997.CapSpeech" alt="Visitor Statistics" /> -->
  <img src="https://img.shields.io/github/stars/WangHelin1997/CapSpeech" alt="GitHub Stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg" />
</p>


## Introduction

🧢 CapSpeech comprises over *10 million machine-annotated* audio-caption pairs and nearly *0.36 million human-annotated* audio-caption pairs. CapSpeech provides a new benchmark including these tasks:

1. **CapTTS**: style-captioned TTS

2. **CapTTS-SE**: text-to-speech synthesis with sound effects

3. **AccCapTTS**: accent-captioned TTS

4. **EmoCapTTS**: emotion-captioned TTS

5. **AgentTTS**: text-to-speech synthesis for chat agent

[Video](https://github.com/user-attachments/assets/b53b7035-d759-43f3-ab80-0ab26748052c)

## Usage
### ⚡ Quick Start  
Explore CapSpeech directly in your browser — no installation needed.  
- 🚀 Live Demo: [🤗 Spaces](https://huggingface.co/spaces/OpenSound/CapSpeech-TTS)

### 🛠️ Local Deployment  
Install and Run CapSpeech locally.  
- 💿 Installation & Usage: [📄 Instrucitons](docs/quick_use.md)

## Development
Please refer to the following documents to prepare the data, train the model, and evaluate its performance.
- [Data Preparation](docs/dataset.md)  
- [Training](docs/training.md)  
- [Evaluation](capspeech/eval/README.md)  

## Main Contributors

- [Helin Wang](https://wanghelin1997.github.io/helinwang/) at Johns Hopkins University
- [Jiarui Hai](https://haidog-yaqub.github.io/) at Johns Hopkins University

## Citation

If you find this work useful, please consider contributing to this repo and cite this work:
```
@misc{wang2025capspeechenablingdownstreamapplications,
      title={CapSpeech: Enabling Downstream Applications in Style-Captioned Text-to-Speech}, 
      author={Helin Wang and Jiarui Hai and Dading Chong and Karan Thakkar and Tiantian Feng and Dongchao Yang and Junhyeok Lee and Laureano Moro Velazquez and Jesus Villalba and Zengyi Qin and Shrikanth Narayanan and Mounya Elhiali and Najim Dehak},
      year={2025},
      eprint={2506.02863},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2506.02863}, 
}
```

## License
All datasets, listening samples, source code, pretrained checkpoints, and the evaluation toolkit are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

This implementation is based on [Parler-TTS](https://github.com/huggingface/parler-tts), [F5-TTS](https://github.com/SWivid/F5-TTS), [SSR-Speech](https://github.com/WangHelin1997/SSR-Speech), [Data-Speech](https://github.com/huggingface/dataspeech), [EzAudio](https://github.com/haidog-yaqub/EzAudio), and [Vox-Profile](https://github.com/tiantiaf0627/vox-profile-release). We appreciate their awesome work.

## 🌟 Like This Project?
If you find this repo helpful or interesting, consider dropping a ⭐ — it really helps and means a lot!
