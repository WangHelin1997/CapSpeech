import torch
import sys, os, pdb
import argparse, logging
import torch.nn.functional as F

from pathlib import Path


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'dialect'))

from wavlm_dialect import WavLMWrapper
from whisper_dialect import WhisperWrapper


# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 


if __name__ == '__main__':

   
    label_list = [
        'East Asia', 'English', 'Germanic', 'Irish', 
        'North America', 'Northern Irish', 'Oceania', 
        'Other', 'Romance', 'Scottish', 'Semitic', 'Slavic', 
        'South African', 'Southeast Asia', 'South Asia', 'Welsh'
    ]
    
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # Define the model
    # Note that ensemble yields the better performance than the single model
    model_path = "YOUR_PATH"
    # Define the model wrapper
    wavlm_model = model = WavLMWrapper(
        pretrain_model="wavlm_large", 
        finetune_method="lora",
        lora_rank=16,
        output_class_num=16,
        freeze_params=False, 
        use_conv_output=True,
        apply_gradient_reversal=False, 
        num_dataset=3
    ).to(device)
    
    whisper_model = WhisperWrapper(
        pretrain_model="whisper_large", 
        finetune_method="lora",
        lora_rank=16,
        output_class_num=16,
        freeze_params=False, 
        use_conv_output=True,
        apply_gradient_reversal=False, 
        num_dataset=11
    ).to(device)
    
    wavlm_model.load_state_dict(torch.load(os.path.join(model_path, f"wavlm_world_dialect.pt"), weights_only=True), strict=False)
    wavlm_model.load_state_dict(torch.load(os.path.join(model_path, f"wavlm_world_dialect_lora.pt")), strict=False)
    
    whisper_model.load_state_dict(torch.load(os.path.join(model_path, f"whisper_world_dialect.pt"), weights_only=True), strict=False)
    whisper_model.load_state_dict(torch.load(os.path.join(model_path, f"whisper_world_dialect_lora.pt")), strict=False)

    wavlm_model.eval()
    whisper_model.eval()
        
    data = torch.zeros([1, 16000]).to(device)
    wavlm_logits, wavlm_embeddings      = wavlm_model(data, return_feature=True)
    whisper_logits, whisper_embeddings  = whisper_model(data, return_feature=True)

    ensemble_logits = (wavlm_logits + whisper_logits) / 2
    ensemble_prob   = F.softmax(ensemble_logits, dim=1)

    pred = label_list[ensemble_prob.argmax(-1)]
    print(pred)

