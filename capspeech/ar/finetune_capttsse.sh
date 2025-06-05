# Please log in to huggingface first

LIBRITTSRMIX_WAV_DIR='' # downloaded capspeech-sedb wav dir
OUTPUT_DIR="./output_finetuning_capttsse/" # output dir, to save checkpoints
TEMPORY_SAVE_TO_DISK="./audio_code_finetuning_capttsse/" # dac codec saved dir
SAVE_TO_DISK="./dataset_finetuning_capttsse/" # huggingface metadata saved dir
WANDB_KEY='' # your wandb key for logging

PRETRAINED_MODEL_PATH="" # your pretrained model path

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

accelerate launch ./training/finetune_capttsse.py \
    --model_name_or_path ${PRETRAINED_MODEL_PATH} \
    --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
    --description_tokenizer_name ${PRETRAINED_MODEL_PATH} \
    --prompt_tokenizer_name ${PRETRAINED_MODEL_PATH} \
    --report_to "wandb" \
    --wandb_key ${WANDB_KEY} \
    --overwrite_output_dir true \
    --train_dataset_name "OpenSound/CapSpeech" \
    --train_split_name "train_SEDB" \
    --eval_dataset_name "OpenSound/CapSpeech" \
    --eval_split_name "test_SEDB" \
    --librittsrmix_dir ${LIBRITTSRMIX_WAV_DIR} \
    --max_eval_samples 96 \
    --per_device_eval_batch_size 32 \
    --target_audio_column_name "audio_path" \
    --description_column_name "caption" \
    --source_column_name "source" \
    --prompt_column_name "text" \
    --max_duration_in_seconds 20 \
    --min_duration_in_seconds 3 \
    --max_text_length 600 \
    --preprocessing_num_workers 32 \
    --do_train true \
    --num_train_epochs 50 \
    --gradient_accumulation_steps 6 \
    --gradient_checkpointing false \
    --per_device_train_batch_size 4 \
    --learning_rate 0.0001 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 50 \
    --logging_steps 20 \
    --freeze_text_encoder true \
    --per_device_eval_batch_size 4 \
    --audio_encoder_per_device_batch_size 24 \
    --dtype "float16" \
    --seed 456 \
    --output_dir ${OUTPUT_DIR} \
    --temporary_save_to_disk ${TEMPORY_SAVE_TO_DISK} \
    --save_to_disk ${SAVE_TO_DISK} \
    --dataloader_num_workers 32 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --group_by_length true