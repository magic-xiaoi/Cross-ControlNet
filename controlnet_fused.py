from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
# from diffusers.loaders import FromOriginalControlnetMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
# from diffusers.models.unet_2d_blocks import (
#     CrossAttnDownBlock2D,
#     DownBlock2D,
#     UNetMidBlock2DCrossAttn,
#     get_down_block,
# )
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
# from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
import numpy as np
import matplotlib.pyplot as plt


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def return_maxvar_feat_intra(controlnet_cond_pose,controlnet_cond_depth):
        
        c =controlnet_cond_pose.size(1)

        A= controlnet_cond_pose
        B= controlnet_cond_depth
        mean_A = torch.mean(A, dim=1, keepdim=True)
        mean_B = torch.mean(B, dim=1, keepdim=True)
        A_demeaned = A - mean_A
        B_demeaned = B - mean_B

        kernel = torch.tensor([[1,2,1],[2,12,2],[1,2,1]], dtype=torch.float32)
        kernel = kernel.view(1,1,3,3)
        kernel = kernel/kernel.sum()
        padding = 3 // 2

        kernel1 = kernel.repeat(c,1,1,1).to(A.device) #(in_Channel,out_channel/groups,h,w)
        A_demeaned = F.conv2d(input=A_demeaned,weight=kernel1,padding=padding,groups=c)
        B_demeaned = F.conv2d(input=B_demeaned,weight=kernel1,padding=padding,groups=c)

        covariance = torch.sum(A_demeaned * B_demeaned, dim=1)
        std_A = torch.sqrt(torch.sum(A_demeaned ** 2, dim=1))
        std_B = torch.sqrt(torch.sum(B_demeaned ** 2, dim=1))
        correlation = covariance / (std_A * std_B)


        std_A = torch.std(A, dim=1, keepdim=True)
        std_B = torch.std(B, dim=1, keepdim=True)
        var_A = torch.var(A , dim=1, keepdim=True)
        var_B = torch.var(B , dim=1, keepdim=True)
        var_A=var_A /torch.sum(var_A)
        var_B=var_B/torch.sum(var_B)

        kernel2 = kernel.repeat(1,1,1,1).to(A.device)
        var_A = F.conv2d(input=var_A,weight=kernel2,padding=padding,groups=1)
        var_B = F.conv2d(input=var_B,weight=kernel2,padding=padding,groups=1)
    
        high_sim_threshold =0.7
        average=(A+B)/2

        fuse_based_on_variance1 = torch.where(var_A >= var_B, A, torch.div(B*std_A,std_B))  
        fuse_based_on_variance2 = torch.where(var_A >= var_B, torch.div(A*std_B,std_A), B)

        # fuse_based_on_variance1 = torch.where(var_A >= var_B, A, mean_A+torch.div((B - mean_B)*std_A,std_B))
        # fuse_based_on_variance2 = torch.where(var_A >= var_B,mean_B+torch.div((A - mean_A)*std_B,std_A), B)


        # Decide which values to take based on the cosine similarity
        fused_tensor1 = torch.where(correlation.unsqueeze(1) > high_sim_threshold, average, fuse_based_on_variance1)
        fused_tensor2 = torch.where(correlation.unsqueeze(1) > high_sim_threshold, average, fuse_based_on_variance2)

        controlnet_cond1=fused_tensor1
        controlnet_cond2=fused_tensor2

        return controlnet_cond1,controlnet_cond2


def return_maxvar_feat_intra_sd1(controlnet_cond_pose,controlnet_cond_depth,window_size=3):
        c =controlnet_cond_pose.size(1)
        
        A= controlnet_cond_pose
        B= controlnet_cond_depth 
        mean_A = torch.mean(A, dim=1, keepdim=True)
        mean_B = torch.mean(B, dim=1, keepdim=True)
        A_demeaned = A - mean_A
        B_demeaned = B - mean_B

        kernel = torch.tensor([[1,2,1],[2,12,2],[1,2,1]], dtype=torch.float32)
        kernel = kernel.view(1,1,3,3)
        kernel = kernel/kernel.sum()
        padding = window_size // 2

        kernel1 = kernel.repeat(c,1,1,1).to(A.device) #(in_Channel,out_channel/groups,h,w)
        A_demeaned = F.conv2d(input=A_demeaned,weight=kernel1,padding=padding,groups=c)
        B_demeaned = F.conv2d(input=B_demeaned,weight=kernel1,padding=padding,groups=c)

        covariance = torch.sum(A_demeaned * B_demeaned, dim=1)
        std_A = torch.sqrt(torch.sum(A_demeaned ** 2, dim=1))
        std_B = torch.sqrt(torch.sum(B_demeaned ** 2, dim=1))
        correlation = covariance / (std_A * std_B)

        var_A = torch.var(A , dim=1, keepdim=True)
        var_B = torch.var(B , dim=1, keepdim=True)
        var_A=var_A /torch.sum(var_A)
        var_B=var_B/torch.sum(var_B)

        kernel2 = kernel.repeat(1,1,1,1).to(A.device)
        var_A = F.conv2d(input=var_A,weight=kernel2,padding=padding,groups=1)
        var_B = F.conv2d(input=var_B,weight=kernel2,padding=padding,groups=1)
  
        high_sim_threshold =0.7
        average=(A+B)/2
   
        # fuse_based_on_variance = torch.where(2.0*var_A >= var_B, A, B) 
        fuse_based_on_variance = torch.where(var_A >= var_B, A, B) 

        fused_tensor = torch.where(correlation.unsqueeze(1) > high_sim_threshold, average, fuse_based_on_variance)

        return fused_tensor


def return_maxvar_feat_intra_sd2(controlnet_cond_pose, controlnet_cond_depth, threshold=0.7):
    """
    Enhanced channel-wise fusion (hard-selection + soft-blending hybrid).
    Channels whose variance ratio exceeds `threshold` are hard-selected;
    the rest are softly blended.
    """
    A = controlnet_cond_pose
    B = controlnet_cond_depth
    
    # Compute per-channel variance
    var_A = torch.var(A, dim=[2, 3], keepdim=True)
    var_B = torch.var(B, dim=[2, 3], keepdim=True)
    
    # Binary mask: 1 where A has larger variance, 0 otherwise
    mask = (var_A > var_B).float()
    
    # Compute relative difference between variances
    diff_ratio = torch.abs(var_A - var_B) / torch.max(var_A, var_B)
    
    # Split into hard and soft regions
    hard_mask = (1 - diff_ratio < threshold).float()
    soft_mask = 1 - hard_mask
    
    # Hard selection: pick the more variant feature
    fused_hard = mask * A + (1 - mask) * B
    
    # Soft blending: variance-weighted average
    fused_soft = (var_A / (var_A + var_B)) * A + (var_B / (var_A + var_B)) * B
    
    # Combine hard and soft results
    return hard_mask * fused_hard + soft_mask * fused_soft


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor

# Adapted from ControlNetModel.forward()
def forwardfused(
        model1,
        model2,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states1: torch.Tensor,
        encoder_hidden_states2: torch.Tensor,
        controlnet_cond_pose: torch.FloatTensor,
        controlnet_cond_depth: torch.FloatTensor,
        conditioning_scale: float=1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        step:int =0,
        mask1:torch.FloatTensor = None,
        mask2:torch.FloatTensor=None,
        thres:float=1/256
    ) -> Union[ControlNetOutput, Tuple]:


        from  unet_utils.attention_processor import register_attention_processor
        register_attention_processor(model1, processor_type="MyAttnProcessor")
        register_attention_processor(model2, processor_type="MyAttnProcessor")

 
        channel_order = model1.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond_pose = torch.flip(controlnet_cond_pose, dims=[1])
            controlnet_cond_depth= torch.flip(controlnet_cond_depth, dims=[1])

        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t_emb1 = model1.time_proj(timesteps)
        t_emb2 = model2.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb1= t_emb1.to(dtype=sample.dtype)
        # print(model1.device, t_emb1.device)
        emb1 = model1.time_embedding(t_emb1, timestep_cond)
        emb2 = model2.time_embedding(t_emb2, timestep_cond)

        aug_emb1 = None
        aug_emb2 = None

        if model1.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if model1.config.class_embed_type == "timestep":
                class_labels1 = model1.time_proj(class_labels)
                class_labels2 = model2.time_proj(class_labels)

            class_emb1 =model1.class_embedding(class_labels1).to(dtype=model1.dtype)
            class_emb2 =model2.class_embedding(class_labels2).to(dtype=model2.dtype)

            emb1 = emb1 + class_emb1
            emb2 = emb2 + class_emb2

        if model1.config.addition_embed_type is not None:
            if model1.config.addition_embed_type == "text":
                aug_emb1 = model1.add_embedding(encoder_hidden_states1)
                aug_emb2 = model2.add_embedding(encoder_hidden_states2)

            elif model1.config.addition_embed_type == "text_time":
             
                text_embeds1 = added_cond_kwargs.get("text_embeds")
                text_embeds2 = added_cond_kwargs.get("text_embeds")

              
                time_ids1 = added_cond_kwargs.get("time_ids")
                time_embeds1 = model1.add_time_proj(time_ids1.flatten())
                time_embeds1 = time_embeds1.reshape((text_embeds1.shape[0], -1))

                add_embeds1 = torch.concat([text_embeds1, time_embeds1], dim=-1)
                add_embeds1 = add_embeds1.to(emb1.dtype)
                aug_emb1 = model1.add_embedding(add_embeds1)


                  
                time_ids2 = added_cond_kwargs.get("time_ids")
                time_embeds2 = model2.add_time_proj(time_ids2.flatten())
                time_embeds2 = time_embeds2.reshape((text_embeds2.shape[0], -1))

                add_embeds = torch.concat([text_embeds2, time_embeds2], dim=-1)
                """这里原本是add_embeds = add_embeds.to(emb.dtype)，代码是错的"""
                add_embeds = add_embeds.to(emb2.dtype)
                aug_emb2 =model2.add_embedding(add_embeds)

        emb1 = emb1 + aug_emb1 if aug_emb1 is not None else emb1
        emb2 = emb2 + aug_emb2 if aug_emb2 is not None else emb2

        sample1 = model1.conv_in(sample)
        sample2 = model2.conv_in(sample)
       
        # print(controlnet_cond_pose.shape)
        controlnet_cond1 = model1.controlnet_cond_embedding(controlnet_cond_pose)
        controlnet_cond2 = model2.controlnet_cond_embedding(controlnet_cond_depth)
        # print(controlnet_cond1.shape)

        sample1,sample2=return_maxvar_feat_intra(sample1,sample2)

        sample1 = sample1 + controlnet_cond1
        sample2 = sample2 + controlnet_cond2


        down_block_res_samples1 = (sample1,)
        down_block_res_samples2= (sample2,)

        current_BasicTransformerBlock=-1
        for downsample_block1,downsample_block2 in zip(model1.down_blocks,model2.down_blocks):
            if hasattr(downsample_block1, "has_cross_attention") and downsample_block1.has_cross_attention:
                res_samples1,res_samples2=(),()
                for i in range(len(downsample_block1.resnets)):
                    current_BasicTransformerBlock=current_BasicTransformerBlock+1

                    # ------------------------------------------------------------------
                    # Under investigation – kept for future experiments
                    # ------------------------------------------------------------------
                    if False and 50 <= step < 4 and 4 <= current_BasicTransformerBlock <= 4:
                        from unet.resnet import ResnetBlock2D

                        # Retrieve intermediate activations from the ResNet blocks
                        out_layers_features1 = ResnetBlock2D.get_out_layers_features(downsample_block1.resnets[i], sample1, emb1)
                        out_layers_features2 = ResnetBlock2D.get_out_layers_features(downsample_block2.resnets[i], sample2, emb2)

                        # Inject features (main modification point)
                        sample1 = ResnetBlock2D.forward_injected(downsample_block1.resnets[i], sample1, out_layers_features1)
                        sample2 = ResnetBlock2D.forward_injected(downsample_block2.resnets[i], sample2, out_layers_features2)
                    else:
                        sample1 = downsample_block1.resnets[i](sample1, emb1)
                        sample2 = downsample_block2.resnets[i](sample2, emb2)

                    from  unet_utils.transformer_2d import forward1 as transformer_2d_forward1
                    from  unet_utils.transformer_2d import forward3 as transformer_2d_forward3
                    from  unet_utils.transformer_2d import forward2 as transformer_2d_forward2

                    sample1_forward1_result = transformer_2d_forward1(
                        downsample_block1.attentions[i],
                        hidden_states=sample1,
                        encoder_hidden_states=encoder_hidden_states1,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        return_dict=False,)
                    sample2_forward1_result = transformer_2d_forward1(
                        downsample_block2.attentions[i],
                        hidden_states=sample2,
                        encoder_hidden_states=encoder_hidden_states2,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        return_dict=False,)


                    if True and 5<=step<=50 and 0<=current_BasicTransformerBlock<=5:
                        from  unet_utils.BasicTransformerBlock import BasicTransformerBlock

                        q1,k1,v1 = BasicTransformerBlock.get_qkv(
                            downsample_block1.attentions[i].transformer_blocks[0],
                            *sample1_forward1_result[0],)
                        
                        q2,k2,v2 = BasicTransformerBlock.get_qkv(
                            downsample_block2.attentions[i].transformer_blocks[0],
                            *sample2_forward1_result[0],)
                        
                        res = int(np.sqrt(q1.shape[1]))
                        mask1_adapted = F.interpolate(mask1, (res, res),mode="bicubic").flatten().to(model1.dtype)
                        mask2_adapted = F.interpolate(mask2, (res, res),mode="bicubic").flatten().to(model2.dtype)
                        
                        import copy
                        mask1_temp=copy.deepcopy(mask1_adapted)
                        mask2_temp=copy.deepcopy(mask2_adapted)

                        mask1_adapted[mask1_temp < thres] = -torch.inf
                        mask1_adapted[mask1_temp >= thres] = 0
                        mask2_adapted[mask2_temp < thres] = -torch.inf
                        mask2_adapted[mask2_temp >= thres] = 0


                        sample1_forward2_reslut = BasicTransformerBlock.forward(
                            downsample_block1.attentions[i].transformer_blocks[0],
                            sample1_forward1_result[0][0],
                            sample1_forward1_result[0][1],
                            *sample1_forward1_result[0][2:],
                            query=q1,
                            key=k1,
                            value=v1)


                        sample2_forward2_reslut = BasicTransformerBlock.forward(
                            downsample_block2.attentions[i].transformer_blocks[0],
                            sample2_forward1_result[0][0],
                            # sample2_forward1_result[0][1],
                            [mask1_adapted,mask2_adapted],
                            *sample2_forward1_result[0][2:],
                            query=q2,
                            key=[k2,k1],
                            value=[v2,v1])

                    
                    else:
                        sample1_forward2_reslut = transformer_2d_forward2(
                            downsample_block1.attentions[i],
                            *sample1_forward1_result[0],)
                        
                        sample2_forward2_reslut = transformer_2d_forward2(
                            downsample_block2.attentions[i],
                            *sample2_forward1_result[0],)
                    

                    sample1=transformer_2d_forward3(
                        downsample_block1.attentions[i],
                        sample1_forward2_reslut,
                        *sample1_forward1_result[1],)[0]
                    sample2=transformer_2d_forward3(
                        downsample_block2.attentions[i],
                        sample2_forward2_reslut,
                        *sample2_forward1_result[1],)[0]
                    
                    res_samples1 = res_samples1 + (sample1,)
                    res_samples2 = res_samples2 + (sample2,)


                if downsample_block1.downsamplers is not None:
                    for downsampler in downsample_block1.downsamplers:
                        sample1 = downsampler(sample1)
                    res_samples1 = res_samples1 + (sample1,)

                if downsample_block2.downsamplers is not None:
                    for downsampler in downsample_block2.downsamplers:
                        sample2 = downsampler(sample2)
                    res_samples2 = res_samples2 + (sample2,)


                sample1,sample2=return_maxvar_feat_intra(sample1,sample2)
                

            else:
                sample1, res_samples1 = downsample_block1(hidden_states=sample1, temb=emb1)
                sample2, res_samples2 = downsample_block2(hidden_states=sample2, temb=emb2)


                sample1,sample2=return_maxvar_feat_intra(sample1,sample2)

            down_block_res_samples1 += res_samples1
            down_block_res_samples2 += res_samples2


        if model1.mid_block is not None:
            # ------------------------------------------------------------------
            # Under investigation – kept for future experiments
            # ------------------------------------------------------------------
            if False and 50<=step<=50:
                from unet.resnet import ResnetBlock2D
                
                out_layers_features1=ResnetBlock2D.get_out_layers_features(model1.mid_block.resnets[0],sample1, emb1)
                out_layers_features2=ResnetBlock2D.get_out_layers_features(model2.mid_block.resnets[0],sample2, emb2)
                
                sample1=ResnetBlock2D.forward_injected(model1.mid_block.resnets[0],sample1,out_layers_features1)
                sample2=ResnetBlock2D.forward_injected(model2.mid_block.resnets[0],sample2,out_layers_features2)
            else:
                sample1 = model1.mid_block.resnets[0](sample1, emb1)
                sample2 = model2.mid_block.resnets[0](sample2, emb2)


            from  unet_utils.transformer_2d import forward1 as transformer_2d_forward1
            from  unet_utils.transformer_2d import forward3 as transformer_2d_forward3
            from  unet_utils.transformer_2d import forward2 as transformer_2d_forward2

            sample1_forward1_result = transformer_2d_forward1(
                model1.mid_block.attentions[0],
                hidden_states=sample1,
                encoder_hidden_states=encoder_hidden_states1,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                return_dict=False,)
            sample2_forward1_result = transformer_2d_forward1(
                model2.mid_block.attentions[0],
                hidden_states=sample2,
                encoder_hidden_states=encoder_hidden_states2,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                return_dict=False,)
            

            if True and 5<=step<=50:
                from  unet_utils.BasicTransformerBlock import BasicTransformerBlock

                q1,k1,v1 = BasicTransformerBlock.get_qkv(
                    model1.mid_block.attentions[0].transformer_blocks[0],
                    *sample1_forward1_result[0],)
                
                q2,k2,v2 = BasicTransformerBlock.get_qkv(
                    model2.mid_block.attentions[0].transformer_blocks[0],
                    *sample2_forward1_result[0],)
                
                res = int(np.sqrt(q1.shape[1]))
                mask1_adapted = F.interpolate(mask1, (res, res),mode="bicubic").flatten().to(model1.dtype)
                mask2_adapted = F.interpolate(mask2, (res, res),mode="bicubic").flatten().to(model2.dtype)

                import copy
                mask1_temp=copy.deepcopy(mask1_adapted)
                mask2_temp=copy.deepcopy(mask2_adapted)

                mask1_adapted[mask1_temp < thres] = -torch.inf
                mask1_adapted[mask1_temp >= thres] = 0
                mask2_adapted[mask2_temp < thres] = -torch.inf
                mask2_adapted[mask2_temp >= thres] = 0


                sample1_forward2_reslut = BasicTransformerBlock.forward(
                    model1.mid_block.attentions[0].transformer_blocks[0],
                    sample1_forward1_result[0][0],
                    sample1_forward1_result[0][1],
                    *sample1_forward1_result[0][2:],
                    query=q1,
                    key=k1,
                    value=v1)

                        
                sample2_forward2_reslut = BasicTransformerBlock.forward(
                    model2.mid_block.attentions[0].transformer_blocks[0],
                    sample2_forward1_result[0][0],
                    # sample2_forward1_result[0][1],
                    [mask1_adapted,mask2_adapted],
                    *sample2_forward1_result[0][2:],
                    query=q2,
                    key=[k2,k1],
                    value=[v2,v1])
                                                        
                    
            else:
                sample1_forward2_reslut = transformer_2d_forward2(
                    model1.mid_block.attentions[0],
                    *sample1_forward1_result[0],)
                
                sample2_forward2_reslut = transformer_2d_forward2(
                    model2.mid_block.attentions[0],
                    *sample2_forward1_result[0],)
            

            sample1=transformer_2d_forward3(
                model1.mid_block.attentions[0],
                sample1_forward2_reslut,
                *sample1_forward1_result[1],)[0]
            sample2=transformer_2d_forward3(
                model2.mid_block.attentions[0],
                sample2_forward2_reslut,
                *sample2_forward1_result[1],)[0]
            

            sample1 = model1.mid_block.resnets[1](sample1, emb1)
            sample2 = model2.mid_block.resnets[1](sample2, emb2)

            sample1,sample2=return_maxvar_feat_intra(sample1,sample2)

        controlnet_down_block_res_samples1 = ()
        controlnet_down_block_res_samples2 = ()
        controlnet_down_block_res_samples3 = ()


        for down_block_res_sample1, controlnet_block1,down_block_res_sample2, controlnet_block2 in zip(down_block_res_samples1, model1.controlnet_down_blocks,down_block_res_samples2, model2.controlnet_down_blocks):
            down_block_res_sample1 = controlnet_block1(down_block_res_sample1)
            down_block_res_sample2 = controlnet_block2(down_block_res_sample2)

            
            down_block_res_sample3=return_maxvar_feat_intra_sd1(down_block_res_sample1,down_block_res_sample2)
            import copy
            down_block_res_sample1,down_block_res_sample2=copy.deepcopy(down_block_res_sample3)
            
            controlnet_down_block_res_samples1 = controlnet_down_block_res_samples1 + (down_block_res_sample1,)
            controlnet_down_block_res_samples2 = controlnet_down_block_res_samples2 + (down_block_res_sample2,)
            controlnet_down_block_res_samples3 = controlnet_down_block_res_samples3 + (down_block_res_sample3,)

        down_block_res_samples1 = controlnet_down_block_res_samples1
        down_block_res_samples2 = controlnet_down_block_res_samples2
        down_block_res_samples3 = controlnet_down_block_res_samples3


        mid_block_res_sample1 = model1.controlnet_mid_block(sample1)
        mid_block_res_sample2 = model2.controlnet_mid_block(sample2)

        mid_block_res_sample3 = return_maxvar_feat_intra_sd2(mid_block_res_sample1,mid_block_res_sample2)
        mid_block_res_sample1,mid_block_res_sample2=copy.deepcopy(mid_block_res_sample3)

        # 6. scaling
        if guess_mode and not model1.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples1 = [sample * conditioning_scale for sample in down_block_res_samples1]
            mid_block_res_sample1 = mid_block_res_sample1 * conditioning_scale
            down_block_res_samples2 = [sample * conditioning_scale for sample in down_block_res_samples2]
            mid_block_res_sample2 = mid_block_res_sample2 * conditioning_scale
            down_block_res_samples3 = [sample * conditioning_scale for sample in down_block_res_samples3]
            mid_block_res_sample3 = mid_block_res_sample3 * conditioning_scale
        
        if model1.config.global_pool_conditions:
            # stop
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)



        if not return_dict:
            # return (down_block_res_samples1, mid_block_res_sample1) #poor
            return (down_block_res_samples1, mid_block_res_sample2) #good
            # return (down_block_res_samples1, mid_block_res_sample3)
            # return (down_block_res_samples2, mid_block_res_sample1) #poor
            # return (down_block_res_samples2, mid_block_res_sample2)
            # return (down_block_res_samples2, mid_block_res_sample3)
            # return (down_block_res_samples3, mid_block_res_sample1) #poor
            # return (down_block_res_samples3, mid_block_res_sample2) #good
            # return (down_block_res_samples3, mid_block_res_sample3)

            
        # return ControlNetOutput(
        #     down_block_res_samples=down_block_res_samples1, mid_block_res_sample=mid_block_res_sample2
        # )
