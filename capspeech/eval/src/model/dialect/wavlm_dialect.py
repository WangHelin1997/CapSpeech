import os
import pdb
import copy
import torch
import argparse
import loralib as lora
import transformers.models.wavlm.modeling_wavlm as wavlm
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.lobes.models.huggingface_transformers.huggingface import make_padding_masks

from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2FeatureExtractor
from transformers import WavLMModel

import sys
from pathlib import Path
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
from revgrad import RevGrad

class WavLMEncoderLayer(nn.Module):
    def __init__(self, layer_idx, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config
        
        if layer_idx > config.num_hidden_layers // 2:
            if self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
                self.feed_forward.intermediate_dense    = lora.Linear(config.hidden_size, config.intermediate_size, r=config.lora_rank)
                self.feed_forward.output_dense          = lora.Linear(config.intermediate_size, config.hidden_size, r=config.lora_rank)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class WavLMEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, layer_idx, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config

        if layer_idx > config.num_hidden_layers // 2:
            if self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
                self.feed_forward.intermediate_dense    = lora.Linear(config.hidden_size, config.intermediate_size, r=config.lora_rank)
                self.feed_forward.output_dense          = lora.Linear(config.intermediate_size, config.hidden_size, r=config.lora_rank)
            

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

   
class WavLMWrapper(nn.Module):
    def __init__(
        self, 
        pretrain_model="wavlm_large", 
        hidden_dim=256,
        finetune_method="lora",
        lora_rank=16,
        freeze_params=True,
        output_class_num=4,
        use_conv_output=True,
        apply_gradient_reversal=False,
        num_dataset=4
    ):
        super(WavLMWrapper, self).__init__()
        # 1. We Load the model first with weights
        if pretrain_model == "wavlm":
            self.backbone_model = WavLMModel.from_pretrained(
                "microsoft/wavlm-base-plus",
                output_hidden_states=True,
            )
        elif pretrain_model == "wavlm_large":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
            self.backbone_model = WavLMModel.from_pretrained(
                "microsoft/wavlm-large",
                output_hidden_states=True,
            )
        self.pretrain_model             = pretrain_model
        self.finetune_method            = finetune_method
        self.apply_gradient_reversal    = apply_gradient_reversal
        self.use_conv_output            = use_conv_output
        
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method        = finetune_method
        self.model_config.lora_rank              = lora_rank
        
        # 3. Config encoder layers with adapter or embedding prompt
        if self.pretrain_model == "wavlm":
            self.backbone_model.encoder.layers = nn.ModuleList(
                [WavLMEncoderLayer(i, self.model_config, has_relative_position_bias=(i == 0)) for i in range(self.model_config.num_hidden_layers)]
            )
        elif self.pretrain_model == "wavlm_large":
            self.backbone_model.encoder.layers = nn.ModuleList(
                [WavLMEncoderLayerStableLayerNorm(i, self.model_config, has_relative_position_bias=(i == 0)) for i in range(self.model_config.num_hidden_layers)]
            )
        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)

        # 5. Freeze the weights
        self.freeze_params = freeze_params
        if self.freeze_params and self.finetune_method != "lora":
            for _, p in self.backbone_model.named_parameters(): p.requires_grad = False
        elif self.freeze_params and self.finetune_method == "lora":
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False
        else:
            for _, p in self.backbone_model.named_parameters(): p.requires_grad = True

        # 6. Downstream models
        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0)
        )

        if self.use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1  # transformer layers + input embeddings
            self.weights = nn.Parameter(torch.ones(num_layers)/num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.weights = nn.Parameter(torch.zeros(num_layers))
        
        if apply_gradient_reversal:
            self.dataset_layer = nn.Sequential(
                RevGrad(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_dataset),
            )
        
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )
        
    def forward(self, x, length=None, return_feature=False):
        # 1. feature extraction and projections
        if self.pretrain_model == "wavlm_large":  
            with torch.no_grad():
                signal, attention_mask = list(), list()
                if length is not None: attention_mask = make_padding_masks(x, wav_len=length/length.max()).to(x.device)
                else: attention_mask = make_padding_masks(x, wav_len=torch.tensor([1]).to(x.device)).to(x.device)

                for idx in range(len(x)):
                    input = self.processor(x[idx], sampling_rate=16_000, return_tensors="pt", padding=True)
                    signal.append(input["input_values"][0].to(x.device))
                signal = torch.stack(signal)

        # 2. get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            length = length.cuda()

        if self.pretrain_model == "wavlm": 
            x = self.backbone_model(
                x, output_hidden_states=True
            ).hidden_states
        else: 
            x = self.backbone_model(
                signal, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            ).hidden_states
        
        # 4. stacked feature
        if self.use_conv_output: stacked_feature = torch.stack(x, dim=0)
        else: stacked_feature = torch.stack(x, dim=0)[1:]
        
        # 5. Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        if self.use_conv_output:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers+1, -1)
        else:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        
        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)
        
        # 6. Pass the weighted average to point-wise 1D Conv
        # B x T x D
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)
        
        # 7. Pooling
        if length is not None:
            mean, std = list(), list()
            for snt_id in range(features.shape[0]):
                # Avoiding padded time steps
                actual_size = length[snt_id]
                mean.append(torch.mean(features[snt_id, 0:actual_size, ...], dim=0))
            features = torch.stack(mean)
        else:
            features = torch.mean(features, dim=1)

        # 8. Output predictions
        # B x D
        predicted = self.out_layer(features)
        if self.apply_gradient_reversal: 
            dataset_predicted = self.dataset_layer(features)
            if return_feature: return predicted, dataset_predicted, features
            return predicted, dataset_predicted
        if return_feature: return predicted, features
        return predicted
    
    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.backbone_model.config.conv_kernel, self.backbone_model.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

def prepare_mask(length, shape, dtype):
    # Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[(torch.arange(mask.shape[0]), length.cpu() - 1)] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask
    
    