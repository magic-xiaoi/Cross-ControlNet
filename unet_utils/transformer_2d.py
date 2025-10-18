from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import deprecate, is_torch_version, logging

class Transformer2DModelOutput(Transformer2DModelOutput):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `Transformer2DModelOutput` from `diffusers.models.transformer_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.modeling_outputs import Transformer2DModelOutput`, instead."
        deprecate("Transformer2DModelOutput", "1.0.0", deprecation_message)
        super().__init__(*args, **kwargs)


def forward1(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Dict[str, torch.Tensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            print("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    if attention_mask is not None and attention_mask.ndim == 2:

        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 1. Input
    if self.is_input_continuous:
        batch_size, _, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
    
    return ((hidden_states,attention_mask,encoder_hidden_states,encoder_attention_mask,timestep,cross_attention_kwargs,class_labels),(batch_size,height,width,residual,inner_dim,return_dict))

def forward2(self,hidden_states,attention_mask,encoder_hidden_states,encoder_attention_mask,timestep,cross_attention_kwargs,class_labels):
    # 2. Blocks
    for block in self.transformer_blocks:
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
        )
    return hidden_states

def forward3(self,hidden_states,batch_size,height,width,residual,inner_dim,return_dict):
    # 3. Output
    if self.is_input_continuous:
        output = self._get_output_for_continuous_inputs(
            hidden_states=hidden_states,
            residual=residual,
            batch_size=batch_size,
            height=height,
            width=width,
            inner_dim=inner_dim,
        )

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
