import torch
import logging
import sys, os, pdb
import torch.nn.functional as F

from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'emotion'))

from wavlm_emotion import WavLMWrapper
from whisper_emotion import WhisperWrapper


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
        'Anger', 
        'Contempt', 
        'Disgust', 
        'Fear', 
        'Happiness', 
        'Neutral', 
        'Sadness', 
        'Surprise', 
        'Other'
    ]

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # Define the model
    # Note that ensemble yields the better performance than the single model
    # Define the model wrapper
    model_path = "model"
    wavlm_model = model = WavLMWrapper(
        pretrain_model="wavlm_large", 
        finetune_method="finetune",
        output_class_num=9,
        freeze_params=True, 
        use_conv_output=True,
        detailed_class_num=17
    ).to(device)
    
    whisper_model = WhisperWrapper(
        pretrain_model="whisper_large", 
        finetune_method="lora",
        lora_rank=16, 
        output_class_num=9,
        freeze_params=True, 
        use_conv_output=True,
        detailed_class_num=17
    ).to(device)
        
    whisper_model.load_state_dict(torch.load(os.path.join(model_path, f"whisper_emotion.pt"), weights_only=True), strict=False)
    whisper_model.load_state_dict(torch.load(os.path.join(model_path, f"whisper_emotion_lora.pt")), strict=False)
    wavlm_model.load_state_dict(torch.load(os.path.join(model_path, f"wavlm_emotion.pt"), weights_only=True), strict=False)

    wavlm_model.eval()
    whisper_model.eval()
    
    # Audio must be 16k Hz
    data = torch.zeros([1, 16000]).to(device)
    whisper_logits, whisper_embedding, _, _, _, _   = whisper_model(
        data, return_feature=True
    )
    wavlm_logits, wavlm_embedding, _, _, _, _       = wavlm_model(
        data, return_feature=True
    )
    
    ensemble_logits = (whisper_logits + wavlm_logits) / 2
    ensemble_prob   = F.softmax(ensemble_logits, dim=1)

    print(ensemble_prob.shape)
    print(whisper_embedding.shape)
    print(wavlm_embedding.shape)
    print(label_list[torch.argmax(ensemble_prob).detach().cpu().item()])

