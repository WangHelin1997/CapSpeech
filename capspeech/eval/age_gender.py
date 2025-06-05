import audeer
import audonnx
import numpy as np

def age_gender_apply(waveform):
    age_labels = ['child', 'teenager', 'young adult', 'middle-aged adult', 'elderly']
    gender_labels = ['female', 'male']
    url = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'
    cache_root = audeer.mkdir('cache')
    model_root = audeer.mkdir('model')
    sampling_rate = 16000
    archive_path = audeer.download_url(url, cache_root, verbose=True)
    audeer.extract_archive(archive_path, model_root)
    model = audonnx.load(model_root)

    result = model(waveform, sampling_rate)
    # Process age
    age_label = result['logits_age'].squeeze() * 100.0
    if age_label <= 12:
        age_label = 'child'
    elif age_label <= 19:
        age_label = 'teenager'
    elif age_label <= 39:
        age_label = 'young adult'
    elif age_label <= 64:
        age_label = 'middle-aged adult'
    else:
        age_label = 'elderly'

    # Process gender
    gender_label = result['logits_gender'].squeeze()
    gender_label = gender_label[:2]  # Remove child
    gender_label = np.argmax(gender_label)

    return age_label, gender_labels[gender_label]
