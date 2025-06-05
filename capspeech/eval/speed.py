from pyannote.audio import Model
from pathlib import Path
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
import torch 
from huggingface_hub import hf_hub_download
import numpy as np

def speed_apply(waveform):
    ratio = 16000/270
    sampling_rate = 16000
    device = "cpu"
    waveform = torch.Tensor(waveform).unsqueeze(0)
    model = Model.from_pretrained(
            Path(hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")),
            strict=False,
        )
    model.to(device)

    pipeline = RegressiveActivityDetectionPipeline(segmentation=model, batch_size=1)
    pipeline.to(torch.device(device))

    device = pipeline._models["segmentation"].device

    res = pipeline({"sample_rate": sampling_rate,
                    "waveform": waveform.to(device).float()})

    speech_duration = sum(map(lambda x: x[0].duration, res["annotation"].itertracks()))     
        
    return speech_duration