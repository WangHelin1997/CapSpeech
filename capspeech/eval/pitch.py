import torch 
import penn

def pitch_apply(waveform):
    hopsize = .01
    fmin = 30.
    fmax = 1000.
    checkpoint = None
    center = 'half-hop'
    interp_unvoiced_at = .065
    sampling_rate = 16000
    penn_batch_size = 4096
    waveform = torch.Tensor(waveform).unsqueeze(0)
    pitch, periodicity = penn.from_audio(
        waveform.float(),
        sampling_rate,
        hopsize=hopsize,
        fmin=fmin,
        fmax=fmax,
        checkpoint=checkpoint,
        batch_size=penn_batch_size,
        center=center,
        interp_unvoiced_at=interp_unvoiced_at,
        gpu=None
        )     
    
    pitch_mean = pitch.mean().cpu().numpy()
    pitch_std = pitch.std().cpu().numpy()

    return pitch_mean, pitch_std