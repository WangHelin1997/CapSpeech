import os
import copy
import torch
import loralib as lora
import transformers.models.whisper.modeling_whisper as whisper

from torch import nn
from transformers.activations import ACT2FN
from transformers import WhisperModel, AutoFeatureExtractor

import sys
from pathlib import Path
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))

class WhisperEncoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = whisper.WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.config = config
        
        if layer_idx > config.encoder_layers // 2:
            if self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
                self.fc1 = lora.Linear(self.embed_dim, config.encoder_ffn_dim, r=config.lora_rank)
                self.fc2 = lora.Linear(config.encoder_ffn_dim, self.embed_dim, r=config.lora_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

   
class WhisperWrapper(nn.Module):
    def __init__(
        self, 
        pretrain_model="wavlm_large", 
        hidden_dim=256,
        finetune_method="lora",
        lora_rank=16,
        freeze_params=True,
        output_class_num=4,
        use_conv_output=True,
        detailed_class_num=17
    ):
        super(WhisperWrapper, self).__init__()
        # 1. We Load the model first with weights
        self.pretrain_model     = pretrain_model
        self.finetune_method    = finetune_method
        self.freeze_params      = freeze_params
        self.use_conv_output    = use_conv_output
        self.lora_rank          = lora_rank
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny", chunk_length=15)
        if self.pretrain_model == "whisper_tiny":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-tiny",
                output_hidden_states=True,
                ignore_mismatched_sizes=True,
                max_source_positions=750,
            )
        elif self.pretrain_model == "whisper_base":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-base",
                output_hidden_states=True,
                ignore_mismatched_sizes=True,
                max_source_positions=750,
            )
        elif self.pretrain_model == "whisper_small":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-small",
                output_hidden_states=True,
                max_source_positions=750,
                ignore_mismatched_sizes=True
            )
        elif self.pretrain_model == "whisper_medium":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-medium",
                output_hidden_states=True,
                ignore_mismatched_sizes=True
            )
        elif self.pretrain_model == "whisper_large":
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v3", chunk_length=15)
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-large-v3",
                output_hidden_states=True,
                ignore_mismatched_sizes=True,
                max_source_positions=750,
            )
        self.embed_positions = copy.deepcopy(self.backbone_model.encoder.embed_positions.weight)
        self.embed_positions.requires_grad = False

        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method        = self.finetune_method
        self.model_config.lora_rank              = self.lora_rank

        if self.finetune_method == "lora":
            # 3. Config encoder layers with adapter or embedding prompt
            # pdb.set_trace()
            self.backbone_model.encoder.layers = nn.ModuleList(
                [WhisperEncoderLayer(self.model_config, layer_idx) for layer_idx in range(self.model_config.encoder_layers)]
            )
            # 4. Load the weights back
            msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        
        # 2. Freeze the weights
        if self.freeze_params and self.finetune_method != "lora":
            for _, p in self.backbone_model.named_parameters(): p.requires_grad = False
        elif self.freeze_params and self.finetune_method == "lora":
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False
        else:
            for name, p in self.backbone_model.named_parameters(): 
                if "decoder" not in name and "conv1" not in name and "conv2" not in name and "embed_positions" not in name: p.requires_grad = True
                else: p.requires_grad = False
        
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
        
        self.emotion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

        self.detailed_out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, detailed_class_num),
        )
        
        self.arousal_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.valence_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.dominance_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, x, length=None, return_feature=False):
                
        # 1. feature extraction and projections
        if length is not None:
            max_audio_len = 15*16000
            # Append to list for feature_extractor to work
            new_x = list()
            for idx in range(len(length)):
                new_x.append(x[idx].detach().cpu().numpy())
            
            # Max length is max audio len in a batch
            features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            features = features.input_features.cuda()
        else:
            max_audio_len = 15*16000
            features = self.feature_extractor(
                x[0].detach().cpu(), 
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            features = features.input_features.cuda()
        
        # 2. get length and mask
        if length is not None:
            length = self._get_feat_extract_output_lengths(length.detach().cpu())
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:750])
        else:
            # Replace positional embeddings
            length = torch.tensor([len(x[0])])
            length = self._get_feat_extract_output_lengths(length)
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:750])
            
        # 3. transformer encoding features
        # compute reduced attention_mask corresponding to feature vectors
        features = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states

        features = torch.stack(features, dim=0)[-1]

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
        
        # Output predictions
        # B x D
        predicted = self.emotion_layer(features)
        detailed_predicted = self.detailed_out_layer(features)
        arousal = self.arousal_layer(features)
        valence = self.valence_layer(features)
        dominance = self.dominance_layer(features)
        if return_feature: return predicted, features, detailed_predicted, arousal, valence, dominance
        return predicted, detailed_predicted, arousal, valence, dominance
        
    # From huggingface
    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = input_lengths // 160
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths
