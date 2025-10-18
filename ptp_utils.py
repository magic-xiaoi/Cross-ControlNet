import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import torch.nn.functional as F
from einops import rearrange
import os
import matplotlib.pyplot as plt
from torchvision import transforms as tfms
from scipy.ndimage import binary_dilation

from controlnet_fused import forwardfused


def view_images(images,results_dir='results', num_rows=1, offset_ratio=0.02,):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)
    if os.path.exists(results_dir)==False:
        os.makedirs(results_dir)
    img_name=len(os.listdir(results_dir))
    pil_img.save(os.path.join(results_dir,str(img_name)+'.png'))


def save_images(images,results_path,num_rows=1, offset_ratio=0.02,):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save(results_path)





def prepare_image(
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height))
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image




@torch.no_grad()
def diffusion_step(model, latents, context, t, guidance_scale, control1=None,control2=None, low_resource=False,step=0,mask1=None,mask2=None,thres=None):

    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        guess_mode=False
        latents = latents.type(model.unet.dtype)
        latents_input = torch.cat([latents] * 2)
        controlnet_latent_model_input = latents_input 

        controlnet_prompt_embeds = context.type(model.unet.dtype)

        if model.controlnet1 is None:
            down_block_res_samples, mid_block_res_sample = None,None
        else:
            controlnet_conditioning_scale =1.0
            down_block_res_samples, mid_block_res_sample = forwardfused(
                        model.controlnet1,
                        model.controlnet2,
                        controlnet_latent_model_input.to(model.controlnet.dtype),
                        t.to(model.controlnet.dtype),
                        encoder_hidden_states1=controlnet_prompt_embeds.to(model.controlnet.dtype),
                        encoder_hidden_states2=controlnet_prompt_embeds.to(model.controlnet.dtype),
                        controlnet_cond_pose=control1.to(model.controlnet.dtype),
                        controlnet_cond_depth=control2.to(model.controlnet.dtype),
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                        step=step,
                        mask1=mask1,
                        mask2=mask2,
                        thres=thres,
                    )
        
        noise_pred = model.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=context,
                    cross_attention_kwargs={},
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t.to(noise_pred.device), latents.to(noise_pred.device))["prev_sample"]
    return latents



def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image




def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents



import PIL.Image

# Input and output types are both PIL images.
# Input is RGB (colorful); output is near-black-and-white.
def colorful_to_bw(img):
    img_array = np.array(img)
    threshold = 50
    luminance = np.mean(img_array, axis=2)

    # Pixels whose mean value is below the threshold are set to white
    img_array[luminance > threshold] = [255, 255, 255]

    new_img = Image.fromarray(img_array)
    return new_img


def pil_to_latents(image,vae,device,dtype):
    if isinstance(image, list):
        latent_list = []
        for img in image:
            init_image = tfms.ToTensor()(img).unsqueeze(0) * 2.0 - 1.0
            init_image = init_image.to(device=device, dtype=dtype)
            init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
            latent_list.append(init_latent_dist)
        return latent_list
    else:
        init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
        init_image = init_image.to(device=device, dtype=dtype)
        init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
        return init_latent_dist


def get_mask(img,vae,device,dtype,thres=1/256,size=(512,512),iterations=2):
    img=img.resize(size)
    new_img=colorful_to_bw(img)
    latent_img = pil_to_latents(new_img,vae,device,dtype)

    latent_img[latent_img >= thres] = 1
    latent_img[latent_img < thres] = 0

    for c in range(latent_img.shape[1]):
        binary_image = latent_img[0, c]
        binary_image = binary_image.detach().cpu().numpy().astype(bool)
        dilated_image = binary_dilation(binary_image, iterations=iterations)
        latent_img[0, c] = torch.from_numpy(dilated_image.astype(np.float32))
    
    # fig,axs = plt.subplots(1,4,figsize=(16,4))
    # for c in range(4):
    #     axs[c].imshow(latent_img[0][c].detach().cpu(),cmap='Greys')
    # plt.show()

    return latent_img.to(dtype)



def create_mask_from_attn_map(token_index, step, threshold, dir):
    # Build file name
    file_name = f"tok_{token_index:02d}_step_{step:02d}.png"
    img_path = os.path.join(dir, file_name)

    # Skip if file does not exist
    if not os.path.exists(img_path):
        print(f"File {img_path} not found, skipped.")
        return None

    # Load image and convert to grayscale
    img = Image.open(img_path).convert("L")
    img_array = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]

    # Generate binary mask by thresholding
    mask = (img_array > threshold).astype(np.float32)

    # Convert NumPy array to PyTorch tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    return mask_tensor


# def create_mask_from_attn_map(token_index, step, threshold, dir, save_dir="zz_attn_masks"):
#     # Ensure the save directory exists
#     os.makedirs(save_dir, exist_ok=True)

#     # Build file name
#     file_name = f"tok_{token_index:02d}_step_{step:02d}.png"
#     img_path = os.path.join(dir, file_name)

#     # Skip if file does not exist
#     if not os.path.exists(img_path):
#         print(f"File {img_path} not found, skipped.")
#         return None

#     # Load image and convert to grayscale
#     img = Image.open(img_path).convert("L")
#     img_array = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]

#     # Generate binary mask by thresholding
#     mask = (img_array > threshold).astype(np.float32)

#     # Convert NumPy array to PyTorch tensor
#     mask_tensor = torch.tensor(mask, dtype=torch.float32)

#     # Resize to 64×64
#     mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
#     mask_tensor = F.interpolate(mask_tensor, size=(64, 64), mode="nearest").squeeze(0).squeeze(0)

#     # Save the mask
#     mask_file_name = f"mask_{token_index:02d}_step_{step+1:02d}_thres_{threshold:.2f}.png"
#     mask_path = os.path.join(save_dir, mask_file_name)
#     mask_img = Image.fromarray((mask_tensor.numpy() * 255).astype(np.uint8))
#     mask_img.save(mask_path)

#     return mask_tensor



@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    negative_prompts: List[str],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
    control1=None,
    control2=None,
    thres=0,
    controller=None,
    token_index=None,
    dir=None,
):
    register_attention_control(model, controller)
    
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        negative_prompts, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    control1= prepare_image(
                image=control1,
                width=512,
                height=512,
                batch_size=batch_size,
                num_images_per_prompt=1,
                device=model.unet.device,
                dtype=model.dtype,
                do_classifier_free_guidance=True,
                guess_mode=False,
            )
    control2 = prepare_image(
                image=control2,
                width=512,
                height=512,
                batch_size=batch_size,
                num_images_per_prompt=1,
                device=model.unet.device,
                dtype=model.dtype,
                do_classifier_free_guidance=True,
                guess_mode=False,
            )
    

    # set timesteps
    extra_set_kwargs = {}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    for step, t in tqdm(enumerate(model.scheduler.timesteps), total=len(model.scheduler.timesteps)):
        if step == 0:
            mask1=None
            mask2=None
        else:
            mask1= create_mask_from_attn_map(token_index=token_index,step=step-1,threshold=thres,dir=dir)
            mask2=1-mask1
            mask1 = mask1.unsqueeze(0).unsqueeze(0)
            mask2 = mask2.unsqueeze(0).unsqueeze(0)
            
        latents = diffusion_step(model, latents, context, t, guidance_scale, control1=control1,control2=control2,low_resource = low_resource,step=step,mask1=mask1,mask2=mask2,thres=thres)

    image = latent2image(model.vae, latents)
  
    return image, latent




def register_attention_control(model, controller):
    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()


    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        # The original forward follows the same logic.
        # This forward() works for both self-attention and cross-attention.
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
            """
            Args:
                hidden_states (_type_): 2-D image feature matrix of shape (batch_size, sequence_length, hidden_size).
                encoder_hidden_states (_type_, optional): Text embedding matrix. If None, self-attention is used.
                attention_mask (_type_, optional): _description_. Defaults to None.
                temb (_type_, optional): _description_. Defaults to None.
            """
            is_cross = encoder_hidden_states is not None
            
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)

            # if is_cross:
            #     print(place_in_unet,": cross_attention_probs",attention_probs.shape)
            # if not is_cross:
            #     print(place_in_unet,": self_attention_probs",attention_probs.shape)


# IMPORTANT: only this line is modified
# Feed attention_probs into the controller, which returns the modified attention_probs
# (is_cross indicates whether this is cross-attention)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward


    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0


    # Retrieve all immediate sub-modules of the UNet.
    # named_children() returns an iterator of (name, module) pairs.
    sub_nets = model.unet.named_children()
    for name, module in sub_nets:
        if "down" in name:
            cross_att_count += register_recr(module, 0, "down")
        elif "up" in name:
            cross_att_count += register_recr(module, 0, "up")
        elif "mid" in name:
            cross_att_count += register_recr(module, 0, "mid")

    # Store the total number of attention layers in the controller
    controller.num_att_layers = cross_att_count
