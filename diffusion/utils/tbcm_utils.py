import numpy as np
import torch
import torch.nn.functional as F


def black_filter(z, z_black, threshold=1.0):
    B = z.shape[0]
    mask = torch.ones(B, dtype=torch.float32, device=z.device)
    mse_list = []

    for i in range(B):
        mse = F.mse_loss(z[i], z_black[i], reduction="mean").item()
        mse_list.append(mse)

        if mse < threshold:
            prob = 1 - (mse / threshold)
            if random.random() < prob:
                mask[i] = 0.0

    return mask, mse_list

def compute_refer_route(num_train_timesteps=1000, shift=1.0, num_inference_steps=20):
    timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
    timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
    sigmas = timesteps / num_train_timesteps
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    sigma_min = sigmas[-1].item()
    sigma_max = sigmas[0].item()
    timesteps = np.linspace(
        sigma_max * num_train_timesteps, sigma_min * num_train_timesteps, num_inference_steps
    )
    sigmas = timesteps / num_train_timesteps
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
    timesteps = sigmas * num_train_timesteps

    t_trig = torch.atan(timesteps[1:]/(1000-timesteps[1:]))
    t_trig = torch.cat([torch.tensor([1.5708]), t_trig], dim=0)
    return t_trig

def sample_ordered_timesteps_from_ref(ref_route):
    device = ref_route.device
    midpoints = ref_route

    samples = [ref_route[0].unsqueeze(0)]

    for i in range(len(midpoints)-1):
        low, high = midpoints[i+1], midpoints[i]
        val = torch.empty(1, device=device).uniform_(low.item(), high.item())
        samples.append(val)

    return torch.cat(samples)

def get_timesteps(weighting_scheme, logit_mean, logit_std):
    if weighting_scheme == "logit_normal_trigflow":
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bs,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=None,
        )
        denoise_timesteps = None
    elif weighting_scheme == "logit_normal_trigflow_ladd":
        indices = torch.randint(0, len(config.scheduler.add_noise_timesteps), (bs,))
        u = torch.tensor([config.scheduler.add_noise_timesteps[i] for i in indices])
        if len(config.scheduler.add_noise_timesteps) == 1:
            # zero-SNR
            denoise_timesteps = torch.tensor([1.57080 for i in indices]).float().to(accelerator.device)
        else:
            denoise_timesteps = u.float().to(accelerator.device)

    return u.float().to(accelerator.device), denoise_timesteps

def sample_ordered_timesteps(
    weighting_scheme: str,
    num_steps: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    descending: bool = True,
    device: str = "cpu",
):
    u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=num_steps,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        ).to(device)
    u_sorted = u.sort(descending=descending).values
    u_sorted[0] = 1.57080
    return u_sorted