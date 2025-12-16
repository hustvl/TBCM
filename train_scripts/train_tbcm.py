# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import getpass
import hashlib
import json
import os
import os.path as osp
import time
import types
import warnings
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from PIL import Image
from termcolor import colored
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning
os.environ["DISABLE_XFORMERS"] = "1"


from diffusion import SCMScheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.model_growth_utils import ModelGrowthInitializer
from diffusion.model.nets.sana_ladd import DiscHeadModel, SanaMSCMDiscriminator
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import clip_grad_norm_, dist, flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer
from diffusion.utils.tbcm_utils import compute_refer_route, sample_ordered_timesteps_from_ref, black_filter
from tools.download import find_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaBlock"


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


@torch.inference_mode()
@torch.no_grad()
def log_validation(accelerator, config, model, logger, step, device, vae=None, init_noise=None, generator=None):
    torch.cuda.empty_cache()
    vis_sampler = config.scheduler.vis_sampler
    model = accelerator.unwrap_model(model).eval()
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(1, 1)
    null_y = torch.load(null_embed_path, map_location="cpu")
    null_y = null_y["uncond_prompt_embeds"].to(device)
    sigma_data = config.scheduler.sigma_data

    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []

    def run_sampling(init_z=None, label_suffix="", vae=None, sampler="dpm-solver"):
        latent_outputs = []
        current_image_logs = []
        for prompt in validation_prompts:
            latents = (
                torch.randn(1, config.vae.vae_latent_dim, latent_size, latent_size, device=device)
                if init_z is None
                else init_z
            ) * sigma_data
            embed = torch.load(
                osp.join(config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"),
                map_location="cpu",
            )
            caption_embs, emb_masks = embed["caption_embeds"].to(device), embed["emb_mask"].to(device)
            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)

            scheduler = SCMScheduler()
            scheduler.set_timesteps(
                num_inference_steps=2,
                max_timesteps=1.57080,
                intermediate_timesteps=1.0,
            )
            timesteps = scheduler.timesteps

            model_kwargs["data_info"].update(
                {"cfg_scale": torch.tensor([config.model.cfg_scale] * latents.shape[0]).to(device)}
            )

            #  sCM MultiStep Sampling Loop:
            for i, t in tqdm(list(enumerate(timesteps[:-1]))):
                timestep = t.expand(latents.shape[0]).to(device)

                # model prediction
                model_pred = sigma_data * model(
                    latents / sigma_data,
                    timestep,
                    caption_embs,
                    **model_kwargs,
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents, denoised = scheduler.step(model_pred, i, t, latents, generator=generator, return_dict=False)

            latent_outputs.append(denoised / sigma_data)

        torch.cuda.empty_cache()
        if vae is None:
            vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(vae_dtype)
        for prompt, latent in zip(validation_prompts, latent_outputs):
            latent = latent.to(vae_dtype)
            samples = vae_decode(config.vae.vae_type, vae, latent)
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            )
            image = Image.fromarray(samples)
            current_image_logs.append({"validation_prompt": prompt + label_suffix, "images": [image]})

        return current_image_logs

    # First run with original noise
    image_logs += run_sampling(init_z=None, label_suffix="", vae=vae, sampler=vis_sampler)

    # Second run with init_noise if provided
    if init_noise is not None:
        init_noise = torch.clone(init_noise).to(device)
        image_logs += run_sampling(init_z=init_noise, label_suffix=" w/ init noise", vae=vae, sampler=vis_sampler)

    formatted_images = []
    for log in image_logs:
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        for image in images:
            formatted_images.append((validation_prompt, np.asarray(image)))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for validation_prompt, image in formatted_images:
                tracker.writer.add_images(validation_prompt, image[None, ...], step, dataformats="NHWC")
        elif tracker.name == "wandb":
            import wandb

            wandb_images = []
            for validation_prompt, image in formatted_images:
                wandb_images.append(wandb.Image(image, caption=validation_prompt, file_type="jpg"))
            tracker.log({"validation": wandb_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    def concatenate_images(image_caption, images_per_row=5, image_format="webp"):
        import io

        images = [log["images"][0] for log in image_caption]
        if images[0].size[0] > 1024:
            images = [image.resize((1024, 1024)) for image in images]

        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights[i : i + images_per_row][0] for i in range(0, len(images), images_per_row))

        new_im = Image.new("RGB", (max_width * images_per_row, total_height))

        y_offset = 0
        for i in range(0, len(images), images_per_row):
            row_images = images[i : i + images_per_row]
            x_offset = 0
            for img in row_images:
                new_im.paste(img, (x_offset, y_offset))
                x_offset += max_width
            y_offset += heights[i]
        webp_image_bytes = io.BytesIO()
        new_im.save(webp_image_bytes, format=image_format)
        webp_image_bytes.seek(0)
        new_im = Image.open(webp_image_bytes)

        return new_im

    if config.train.local_save_vis:
        file_format = "webp"
        local_vis_save_path = osp.join(config.work_dir, "log_vis")
        os.umask(0o000)
        os.makedirs(local_vis_save_path, exist_ok=True)
        concatenated_image = concatenate_images(image_logs, images_per_row=5, image_format=file_format)
        save_path = (
            osp.join(local_vis_save_path, f"vis_{step}.{file_format}")
            if init_noise is None
            else osp.join(local_vis_save_path, f"vis_{step}_w_init.{file_format}")
        )
        concatenated_image.save(save_path)

    model.train()
    del vae
    flush()
    return image_logs


def train(
    config,
    args,
    accelerator,
    model,
    model_ema,
    optimizer_G,
    optimizer_D,
    lr_scheduler,
    train_dataloader,
    logger,
    pretrained_model,
    disc,
):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model, max_frames_to_save=100)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0

    # Cache Dataset for BatchSampler
    if args.caching and config.model.multi_scale:
        caching_start = time.time()
        logger.info(
            f"Start caching your dataset for batch_sampler at {cache_file}. \n"
            f"This may take a lot of time...No training will launch"
        )
        train_dataloader.batch_sampler.sampler.set_start(max(train_dataloader.batch_sampler.exist_ids, 0))
        for index, _ in enumerate(train_dataloader):
            accelerator.wait_for_everyone()
            if index % 2000 == 0:
                logger.info(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                print(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
            if (time.time() - caching_start) / 3600 > 3.7:
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
                accelerator.wait_for_everyone()
                break
            if len(train_dataloader.batch_sampler.cached_idx) == len(train_dataloader) - 1000:
                logger.info(
                    f"Saving rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
            continue
        accelerator.wait_for_everyone()
        print(f"Saving rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
        json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
        return

    black_latent = torch.load("./prepare/black_latent.pt", map_location="cpu")
    black_latent = black_latent.to(accelerator.device)

    sigma_data = config.scheduler.sigma_data
    uncond_y = pretrained_model.y_embedder.y_embedding

    total_step = 0

    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = (
            train_dataloader.batch_sampler.sampler
            if (num_replicas > 1 or config.model.multi_scale)
            else train_dataloader.sampler
        )
        sampler.set_epoch(epoch)
        sampler.set_start(max((skip_step - 1) * config.train.train_batch_size, 0))
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        lm_time_all = 0
        vae_time_all = 0
        model_time_all = 0
        for step, batch in enumerate(train_dataloader):
            # image, json_info, key = batch
            data_time_all += time.time() - data_time_start
            data_info = {}

            lm_time_start = time.time()
            if load_text_feat:
                y = batch[1]  # bs, 1, N, C
                y_mask = batch[2]  # bs, 1, 1, N
            else:
                if "T5" in config.text_encoder.text_encoder_name:
                    with torch.no_grad():
                        txt_tokens = tokenizer(
                            batch["text"], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                        ).to(accelerator.device)
                        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                        y_mask = txt_tokens.attention_mask[:, None, None]
                elif (
                    "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name
                ):
                    with torch.no_grad():
                        if not config.text_encoder.chi_prompt:
                            max_length_all = config.text_encoder.model_max_length
                            prompt = batch["text"]
                        else:
                            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                            prompt = [chi_prompt + i for i in batch["text"]]
                            num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
                            max_length_all = (
                                num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
                            )  # magic number 2: [bos], [_]
                        txt_tokens = tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=max_length_all,
                            truncation=True,
                            return_tensors="pt",
                        ).to(accelerator.device)
                        select_index = [0] + list(
                            range(-config.text_encoder.model_max_length + 1, 0)
                        )  # first bos and end N-1
                        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None][
                            :, :, select_index
                        ]
                        y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]
                else:
                    print("error")
                    exit()

            # Sample a random timestep for each image
            bs = y.shape[0]

            ref_route = compute_refer_route(shift=2.5, num_inference_steps=config.scheduler.route_steps).to(accelerator.device)
            denoise_route = sample_ordered_timesteps_from_ref(ref_route)

            grad_norm = None
            lm_time_all += time.time() - lm_time_start
            model_time_start = time.time()
            model_kwargs = dict(y=y, mask=y_mask, data_info=data_info)

            if config.model.cfg_embed:
                config.train.scm_cfg_scale = (
                    config.train.scm_cfg_scale
                    if isinstance(config.train.scm_cfg_scale, list)
                    else [config.train.scm_cfg_scale]
                )
                # sample cfg scales
                scm_cfg_scale = torch.tensor(
                    np.random.choice(config.train.scm_cfg_scale, size=bs, replace=True),
                    device=accelerator.device,
                )
                data_info["cfg_scale"] = scm_cfg_scale

            if total_step % config.train.gradient_accumulation_steps == 0:
                optimizer_G.zero_grad()

            model.train()

            if config.train.scm_loss:
                all_preds = []
                all_dxt_dt = []
                all_t = []
                x_t = torch.randn(bs, config.vae.vae_latent_dim, latent_size, latent_size, device=accelerator.device) * sigma_data
                with torch.no_grad():
                    if config.train.scm_cfg_scale[0] > 1 and config.model.cfg_embed:
                        uncond_y_repeat = uncond_y.repeat(y.shape[0], 1, 1, 1)
                        cfg_y = torch.cat([uncond_y_repeat, y], dim=0)
                        cfg_y_mask = torch.cat([y_mask, y_mask], dim=0)
                        cfg_model_kwargs = dict(y=cfg_y, mask=cfg_y_mask)
                    for i, ti in enumerate(denoise_route):
                        all_preds.append(x_t.detach())
                        t = torch.tensor(ti.expand(x_t.shape[0]).view(-1, 1, 1, 1), device=x_t.device)
                        all_t.append(t)
                        if config.train.scm_cfg_scale[0] > 1 and config.model.cfg_embed:
                            cfg_x_t = torch.cat([x_t, x_t], dim=0)
                            cfg_t = torch.cat([t, t], dim=0)

                            cfg_pretrain_pred = pretrained_model(
                                cfg_x_t / sigma_data, cfg_t.flatten(), **cfg_model_kwargs
                            )
                            cfg_dxt_dt = sigma_data * cfg_pretrain_pred

                            dxt_dt_uncond, dxt_dt = cfg_dxt_dt.chunk(2)

                            scm_cfg_scale = scm_cfg_scale.view(-1, 1, 1, 1)
                            dxt_dt = dxt_dt_uncond + scm_cfg_scale * (dxt_dt - dxt_dt_uncond)
                        else:
                            pretrain_pred = pretrained_model(x_t / sigma_data, t.flatten(), **model_kwargs)
                            dxt_dt = sigma_data * pretrain_pred
                        all_dxt_dt.append(dxt_dt.detach())

                        if i == len(denoise_route) - 1:
                            dt = denoise_route[i]
                        else:
                            dt = denoise_route[i] - denoise_route[i+1]
                        x_t = torch.cos(dt) * x_t - torch.sin(dt) * dxt_dt

                x0 = x_t

                black_z = black_latent.expand(x0.shape[0], -1, -1, -1)
                black_mask, _mse_list = black_filter(x0 / sigma_data, black_z)

                valid_len = len(all_t) - 1
                t = torch.cat(all_t[1:], dim=0)
                x_t = torch.cat(all_preds[1:], dim=0)
                dxt_dt = torch.cat(all_dxt_dt[1:], dim=0)

                y_expanded = y.repeat(valid_len, 1, 1, 1)
                y_mask_expanded = y_mask.repeat(valid_len, 1, 1, 1)
                black_mask_expand = black_mask.repeat(valid_len,)
                data_info_expanded = {**data_info, "cfg_scale": scm_cfg_scale.squeeze(1).squeeze(1).squeeze(1).repeat(valid_len)}

                v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
                v_t = torch.cos(t) * torch.sin(t)

                def model_wrapper(scaled_x_t, t):
                    pred, logvar = accelerator.unwrap_model(model)(
                        scaled_x_t, t.flatten(), y=y_expanded, mask=y_mask_expanded, data_info=data_info_expanded, return_logvar=True, jvp=True
                    )
                    return pred, logvar

                # Adapt from https://github.com/xandergos/sCM-mnist/blob/master/train_consistency.py
                with torch.no_grad():
                    F_theta, F_theta_grad, logvar = torch.func.jvp(
                        model_wrapper, (x_t / sigma_data, t), (v_x, v_t), has_aux=True
                    )

                F_theta, logvar = model(
                    x_t / sigma_data,
                    t.flatten(),
                    y=y_expanded,
                    mask=y_mask_expanded,
                    data_info=data_info_expanded,
                    return_logvar=True,
                    jvp=False,
                )

                logvar = logvar.view(-1, 1, 1, 1)
                F_theta_grad = F_theta_grad.detach()
                F_theta_minus = F_theta.detach()

                # Warmup steps
                r = min(1, global_step / config.train.tangent_warmup_steps)

                progress = (global_step - config.train.start_calmdown_step) / config.train.tangent_calmdown_steps
                progress = min(max(progress, 0.0), 1.0)
                r = r * (1 - progress) + config.train.final_r * progress

                # Calculate gradient g using JVP rearrangement
                g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
                second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
                g = g + second_term

                # Tangent normalization
                g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
                g = g / (g_norm + 0.1)  # 0.1 is the constant c, can be modified but 0.1 was used in the paper

                sigma = torch.tan(t) * sigma_data
                weight = 1 / sigma

                l2_loss = torch.square(F_theta - F_theta_minus - g)

                # Calculate loss with normalization factor
                loss = (weight / torch.exp(logvar)) * l2_loss + logvar

                loss = loss * black_mask_expand.view(-1, 1, 1, 1)
                loss = loss.mean()

                loss_no_logvar = weight * torch.square(F_theta - F_theta_minus - g)
                loss_no_logvar = loss_no_logvar.mean()
                loss_no_weight = l2_loss.mean()
                g_norm = g_norm.mean()
            
            total_loss = config.train.scm_lambda * loss
            total_loss = total_loss / config.train.gradient_accumulation_steps

            accelerator.backward(total_loss)

            total_step += 1

            if total_step % config.train.gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                    if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()):
                        optimizer_G.zero_grad(set_to_none=True)
                        logger.warning("NaN or Inf detected in grad_norm, skipping iteration...")
                        continue

                optimizer_G.step()
                lr_scheduler.step()
                optimizer_G.zero_grad(set_to_none=True)

            model_time_all += time.time() - model_time_start

            if total_step % config.train.gradient_accumulation_steps == 0:
                lr = lr_scheduler.get_last_lr()[0]
                logs = {}
                if config.train.scm_loss:
                    logs.update({args.loss_report_name: accelerator.gather(loss).mean().item()})
                    logs.update({"loss_no_logvar": accelerator.gather(loss_no_logvar).mean().item()})
                    logs.update({"loss_no_weight": accelerator.gather(loss_no_weight).mean().item()})
                    logs.update({"g_norm": accelerator.gather(g_norm).mean().item()})
                    logs.update({"total_loss": accelerator.gather(total_loss).mean().item()})
                if grad_norm is not None:
                    logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
                log_buffer.update(logs)
                if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                    accelerator.wait_for_everyone()
                    t = (time.time() - last_tic) / config.train.log_interval
                    t_d = data_time_all / config.train.log_interval
                    t_m = model_time_all / config.train.log_interval
                    t_lm = lm_time_all / config.train.log_interval
                    t_vae = vae_time_all / config.train.log_interval
                    avg_time = (time.time() - time_start) / (step + 1)
                    eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                    eta_epoch = str(
                        datetime.timedelta(
                            seconds=int(
                                avg_time
                                * (
                                    train_dataloader_len
                                    - sampler.step_start // config.train.train_batch_size
                                    - step
                                    - 1
                                )
                            )
                        )
                    )
                    log_buffer.average()

                    current_step = (
                        global_step - sampler.step_start // config.train.train_batch_size
                    ) % train_dataloader_len
                    current_step = train_dataloader_len if current_step == 0 else current_step
                    info = (
                        f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                        f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, "
                        f"lm:{t_lm:.3f}, vae:{t_vae:.3f}, lr:{lr:.3e}, r: {r},"
                    )
                    info += (
                        f"s:({model.module.h}, {model.module.w}), "
                        if hasattr(model, "module")
                        else f"s:({model.h}, {model.w}), "
                    )

                    info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                    last_tic = time.time()
                    log_buffer.clear()
                    data_time_all = 0
                    model_time_all = 0
                    lm_time_all = 0
                    vae_time_all = 0
                    if accelerator.is_main_process:
                        logger.info(info)

                logs.update(lr=lr)
                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)

                global_step += 1
                if loss_nan_timer > 20:
                    raise ValueError("Loss is NaN too much times. Break here.")
                if global_step % config.train.save_model_steps == 0:
                    if accelerator.is_main_process:
                        os.umask(0o000)
                        ckpt_saved_path = save_checkpoint(
                            osp.join(config.work_dir, "checkpoints"),
                            epoch=epoch,
                            step=global_step,
                            model=accelerator.unwrap_model(model),
                            optimizer=optimizer_G,
                            lr_scheduler=lr_scheduler,
                            generator=generator,
                            add_symlink=True,
                        )

                        if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                            online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                            os.makedirs(online_metric_monitor_dir, exist_ok=True)
                            with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                                f.write(osp.join(config.work_dir, "config.py") + "\n")
                                f.write(ckpt_saved_path)

                if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
                    if accelerator.is_main_process:
                        if validation_noise is not None:
                            log_validation(
                                accelerator=accelerator,
                                config=config,
                                model=model,
                                logger=logger,
                                step=global_step,
                                device=accelerator.device,
                                vae=vae,
                                init_noise=validation_noise,
                                generator=torch.Generator(device="cuda").manual_seed(0),
                            )
                        else:
                            log_validation(
                                accelerator=accelerator,
                                config=config,
                                model=model,
                                logger=logger,
                                step=global_step,
                                device=accelerator.device,
                                vae=vae,
                            )

                # avoid dead-lock of multiscale data batch sampler
                # for internal, refactor dataloader logic to remove the ad-hoc implementation
                if (
                    config.model.multi_scale
                    and (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step) < 30
                ):
                    # global_step = epoch * train_dataloader_len
                    global_step = (
                        (global_step + train_dataloader_len - 1) // train_dataloader_len
                    ) * train_dataloader_len + 1
                    logger.info("Early stop current iteration")
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    break

                data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # os.umask(0o000)
                ckpt_saved_path = save_checkpoint(
                    osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    step=global_step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer_G,
                    lr_scheduler=lr_scheduler,
                    generator=generator,
                    add_symlink=True,
                )

                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)


@pyrallis.wrap()
def main(cfg: SanaConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time
    global load_vae_feat, load_text_feat, validation_noise, text_encoder, tokenizer, model_weight_dtype
    global max_length, validation_prompts, latent_size, valid_prompt_embed_suffix, null_embed_path
    global image_size, cache_file, total_steps, vae_dtype

    config = cfg
    args = cfg

    training_start_time = time.time()
    load_from = True
    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=config.train.resume_lr_scheduler,
        )

    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.train.use_fsdp:
        init_train = "FSDP"
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
    else:
        init_train = "DDP"
        fsdp_plugin = None

    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[init_handler],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb

            wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)

    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")
    cluster = os.environ.get("CLUSTER", "cs")
    if cluster == "cs":
        config.train.early_stop_hours = 3.9
    elif cluster == "nrt":
        config.train.early_stop_hours = 1.9
    image_size = config.model.image_size
    latent_size = int(image_size) // config.vae.vae_downsample_rate
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    max_length = config.text_encoder.model_max_length
    model_weight_dtype = get_weight_dtype(config.model.mixed_precision)
    vae = None
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    validation_noise = (
        torch.randn(
            1,
            config.vae.vae_latent_dim,
            latent_size,
            latent_size,
            device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        if getattr(config.train, "deterministic_validation", False)
        else None
    )
    if not config.data.load_vae_feat:
        vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(vae_dtype)
    tokenizer = text_encoder = None
    if not config.data.load_text_feat:
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name, device=accelerator.device
        )
        text_embed_dim = text_encoder.config.hidden_size
    else:
        text_embed_dim = config.text_encoder.caption_channels
    config.text_encoder.caption_channels = text_embed_dim

    logger.info(f"vae type: {config.vae.vae_type}, path: {config.vae.vae_pretrained}, weight_dtype: {vae_dtype}")
    if config.text_encoder.chi_prompt:
        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
        logger.info(f"Complex Human Instruct: {chi_prompt}")

    os.makedirs(config.train.null_embed_root, exist_ok=True)
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{config.text_encoder.text_encoder_name}_{max_length}token_{text_embed_dim}.pth",
    )
    if config.train.visualize and len(config.train.validation_prompts):
        # preparing embeddings for visualization. We put it here for saving GPU memory
        valid_prompt_embed_suffix = f"{max_length}token_{config.text_encoder.text_encoder_name}_{text_embed_dim}.pth"
        validation_prompts = config.train.validation_prompts
        skip = True
        if config.text_encoder.chi_prompt:
            uuid_chi_prompt = hashlib.sha256(chi_prompt.encode()).hexdigest()
        else:
            uuid_chi_prompt = hashlib.sha256(b"").hexdigest()
        config.train.valid_prompt_embed_root = osp.join(config.train.valid_prompt_embed_root, uuid_chi_prompt)
        Path(config.train.valid_prompt_embed_root).mkdir(parents=True, exist_ok=True)

        if config.text_encoder.chi_prompt:
            # Save complex human instruct to a file
            chi_prompt_file = osp.join(config.train.valid_prompt_embed_root, "chi_prompt.txt")
            with open(chi_prompt_file, "w", encoding="utf-8") as f:
                f.write(chi_prompt)

        for prompt in validation_prompts:
            prompt_embed_path = osp.join(
                config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
            )
            if not (osp.exists(prompt_embed_path) and osp.exists(null_embed_path)):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break
        if accelerator.is_main_process and not skip:
            if config.data.load_text_feat and (tokenizer is None or text_encoder is None):
                logger.info(f"Loading text encoder and tokenizer from {config.text_encoder.text_encoder_name} ...")
                tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name)

            for prompt in validation_prompts:
                prompt_embed_path = osp.join(
                    config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
                )
                if "T5" in config.text_encoder.text_encoder_name:
                    txt_tokens = tokenizer(
                        prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(accelerator.device)
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
                    caption_emb_mask = txt_tokens.attention_mask
                elif (
                    "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name
                ):
                    if not config.text_encoder.chi_prompt:
                        max_length_all = config.text_encoder.model_max_length
                    else:
                        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                        prompt = chi_prompt + prompt
                        num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
                        max_length_all = (
                            num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
                        )  # magic number 2: [bos], [_]

                    txt_tokens = tokenizer(
                        prompt,
                        max_length=max_length_all,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][
                        :, select_index
                    ]
                    caption_emb_mask = txt_tokens.attention_mask[:, select_index]
                else:
                    raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")

                torch.save({"caption_embeds": caption_emb, "emb_mask": caption_emb_mask}, prompt_embed_path)

            null_tokens = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            if "T5" in config.text_encoder.text_encoder_name:
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            elif "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name:
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            else:
                raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")
            torch.save(
                {"uncond_prompt_embeds": null_token_emb, "uncond_prompt_embeds_mask": null_tokens.attention_mask},
                null_embed_path,
            )
            if config.data.load_text_feat:
                del tokenizer
                del text_encoder
            del null_token_emb
            del null_tokens
            flush()

    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"

    # 1. build scheduler
    predict_info = ""
    if config.scheduler.weighting_scheme in ["logit_normal", "mode", "logit_normal_trigflow"]:
        predict_info += (
            f"flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}, "
        )
    logger.info(predict_info)

    # 2. build models
    # student
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        logvar=config.model.logvar,
        cfg_embed=config.model.cfg_embed,
        cfg_embed_scale=config.model.cfg_embed_scale,
        lr_scale=config.train.lr_scale,
        **model_kwargs,
    ).train()

    # teacher
    teacher_model_kwargs = model_init_config(config, latent_size=latent_size)
    teacher_model_kwargs.update({"cross_attn_type": "flash"})
    pretrained_model = build_model(
        config.model.teacher if config.model.teacher else config.model.model,
        config.train.grad_checkpointing,
        use_fp32_attention=False,
        **teacher_model_kwargs,
    ).eval()
    pretrained_model.requires_grad_(False)

    if config.train.ema_update:
        model_ema = deepcopy(model).eval()
    else:
        model_ema = None

    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )
    # 2-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from
    if config.model.load_from is not None and load_from:
        # load student model
        _, missing, unexpected, _ = load_checkpoint(
            config.model.load_from,
            model,
            model_ema=model_ema,
            load_ema = False,
            null_embed_path=null_embed_path,
        )
        _, teacher_missing, teacher_unexpected, _ = load_checkpoint(
            config.model.teacher_model,
            pretrained_model,
            model_ema=None,
            load_ema=False,
            null_embed_path=null_embed_path,
        )

        logger.warning(colored(f"Missing keys: {missing}", "red"))
        logger.warning(colored(f"Unexpected keys: {unexpected}", "red"))
        logger.warning(colored(f"Teacher Missing keys: {teacher_missing}", "red"))
        logger.warning(colored(f"Teacher Unexpected keys: {teacher_unexpected}", "red"))

    # 2-2. model growth
    if config.model_growth is not None:
        assert config.model.load_from is None
        model_growth_initializer = ModelGrowthInitializer(model, config.model_growth)
        model = model_growth_initializer.initialize(
            strategy=config.model_growth.init_strategy, **config.model_growth.init_params
        )

    if config.train.ema_update:
        ema_update(model_ema, model, 0.0)
    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # 3. build dataloader
    config.data.data_dir = config.data.data_dir if isinstance(config.data.data_dir, list) else [config.data.data_dir]
    config.data.data_dir = [
        data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for data in config.data.data_dir
    ]
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dataset = build_dataset(
        asdict(config.data),
        resolution=image_size,
        aspect_ratio_type=config.model.aspect_ratio_type,
        real_prompt_ratio=config.train.real_prompt_ratio,
        max_length=max_length,
        config=config,
        caption_proportion=config.data.caption_proportion,
        sort_dataset=config.data.sort_dataset,
        vae_downsample_rate=config.vae.vae_downsample_rate,
    )
    if config.model.multi_scale:
        drop_last = True
        uuid = hashlib.sha256("-".join(config.data.data_dir).encode()).hexdigest()[:8]
        cache_dir = osp.expanduser(f"~/.cache/_wids_batchsampler_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_pattern = (
            f"{cache_dir}/{getpass.getuser()}-{uuid}-sort_dataset{config.data.sort_dataset}"
            f"-hq_only{config.data.hq_only}-valid_num{config.data.valid_num}"
            f"-aspect_ratio{len(dataset.aspect_ratio)}-droplast{drop_last}"
            f"dataset_len{len(dataset)}"
        )
        cache_file = f"{base_pattern}-num_replicas{num_replicas}-rank{rank}"
        for i in config.data.data_dir:
            cache_file += f"-{i}"
        cache_file += ".json"

        sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        batch_sampler = AspectRatioBatchSampler(
            sampler=sampler,
            dataset=dataset,
            batch_size=config.train.train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=drop_last,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.data.valid_num,
            hq_only=config.data.hq_only,
            cache_file=cache_file,
            caching=args.caching,
            clipscore_filter_thres=args.data.del_img_clip_thr,
        )
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.train.num_workers)
        train_dataloader_len = len(train_dataloader)
        logger.info(f"rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
    else:
        sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        train_dataloader = build_dataloader(
            dataset,
            num_workers=config.train.num_workers,
            batch_size=config.train.train_batch_size,
            shuffle=False,
            sampler=sampler,
        )
        train_dataloader_len = len(train_dataloader)
    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)
    load_text_feat = getattr(train_dataloader.dataset, "load_text_feat", False)

    # 4. build optimizer and lr scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer_G = build_optimizer(model, config.train.optimizer)

    # print learning rates
    if accelerator.is_main_process and config.train.show_gradient:
        logger.info("Learning rates for different layers:")
        logger.info("Generator learning rates:")
        for group in optimizer_G.param_groups:
            if "name" in group:
                logger.info(f"Layer: {group['name']}, Learning rate: {group['lr']:.8f}")
            else:
                logger.info(f"Layer: unnamed, Learning rate: {group['lr']:.8f}")

    lr_scheduler = build_lr_scheduler(config.train, optimizer_G, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, ffn type: {config.model.ffn_type}, "
        f"text encoder: {config.text_encoder.text_encoder_name}, captions: {config.data.caption_proportion}, precision: {config.model.mixed_precision}"
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except Exception as e:
            logger.error(f"Failed to initialize trackers: {e}")
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = train_dataloader_len * config.train.num_epochs
    complete_state_dict = {}

    # Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        _, teacher_missing, teacher_unexpected, _ = load_checkpoint(
            config.model.teacher_model,
            pretrained_model,
            model_ema=None,
            load_ema=config.model.resume_from.get("load_ema", False),
            null_embed_path=null_embed_path,
        )
        logger.warning(colored(f"Teacher Missing keys: {teacher_missing}", "red"))
        logger.warning(colored(f"Teacher Unexpected keys: {teacher_unexpected}", "red"))
        
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                config.model.resume_from["resume_optimizer"] = True
                config.model.resume_from["resume_lr_scheduler"] = True
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["resume_optimizer"] = config.train.load_from_optimizer
                config.model.resume_from["resume_lr_scheduler"] = config.train.load_from_lr_scheduler
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            _, missing, unexpected, _ = load_checkpoint(
                **config.model.resume_from,
                model=model,
                model_ema=model_ema,
                optimizer=optimizer_G,
                lr_scheduler=lr_scheduler,
                null_embed_path=null_embed_path,
            )
            logger.warning(colored(f"Generator Missing keys: {missing}", "red"))
            logger.warning(colored(f"Generator Unexpected keys: {unexpected}", "red"))

            disc_ckpt_path = config.model.resume_from["checkpoint"].replace(
                ".pth", f"_{config.train.suffix_checkpoints}.pth"
            )
            if osp.exists(disc_ckpt_path):
                checkpoint = find_model(disc_ckpt_path)
                heads_state = checkpoint.get("state_dict", checkpoint)

                heads_state = {k: v for k, v in heads_state.items() if not k.startswith("transformer.")}
                complete_state_dict.update(heads_state)

                if optimizer_D is not None and "optimizer" in checkpoint:
                    try:
                        optimizer_D.load_state_dict(checkpoint["optimizer"])
                    except Exception as e:
                        logger.warning(colored(f"Skipping discriminator optimizer resume: {e}", "red"))

            path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass

    # resume randomise
    set_random_seed((start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"]))
    logger.info(f'Set seed: {(start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"])}')

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, pretrained_model = accelerator.prepare(model, pretrained_model)
    optimizer_G, lr_scheduler = accelerator.prepare(optimizer_G, lr_scheduler)

    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        model_ema=model_ema,
        optimizer_G=optimizer_G,
        optimizer_D=None,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        logger=logger,
        pretrained_model=pretrained_model,
        disc=None,
    )


if __name__ == "__main__":

    main()
