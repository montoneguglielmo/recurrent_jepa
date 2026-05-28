import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def setup_device(device: str = "auto") -> torch.device:
    """Set up the compute device. Options: 'auto', 'cuda', or 'cpu'."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    return device

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    stats = {'mean': mean, 'std': std}
    return normalizer, stats

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")
            
            
            
class CEM:
    """
    Cross-Entropy Method optimizer.
 
    Optimizes a flat vector z ∈ R^dim by iteratively sampling from
    N(μ, σ²), evaluating a cost function, and refitting to the elite set.
 
    Parameters
    ----------
    dim          : int   — dimensionality of the search vector
    n_samples    : int   — number of candidates per iteration
    n_elite      : int   — number of top candidates to refit from
    n_iters      : int   — number of CEM iterations
    init_std     : float — initial standard deviation
    min_std      : float — floor on std to avoid premature collapse
    momentum     : float — exponential smoothing on μ/σ updates (0 = no momentum)
    device       : str   — torch device
    """
 
    def __init__(
        self,
        dim: int,
        n_samples: int = 200,
        n_elite: int = 20,
        n_iters: int = 5,
        init_std: float = 1.0,
        min_std: float = 0.01,
        momentum: float = 0.1,
        device: str = "cuda",
    ):
        self.dim = dim
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.init_std = init_std
        self.min_std = min_std
        self.momentum = momentum
        self.device = device
 
    @torch.no_grad()
    def optimize(self, cost_fn, mu=None, sigma=None):
        """
        Run CEM to minimize cost_fn.
 
        Parameters
        ----------
        cost_fn : callable(z: (N, dim)) -> (N,) costs
            Evaluates N candidate solutions in parallel, returns scalar cost each.
        mu      : (dim,) optional initial mean
        sigma   : (dim,) optional initial std
 
        Returns
        -------
        best_z : (dim,) — the best solution found (lowest cost elite mean)
        mu     : (dim,) — final distribution mean (useful for warm-starting)
        sigma  : (dim,) — final distribution std
        """
        if mu is None:
            mu = torch.zeros(self.dim, device=self.device)
        if sigma is None:
            sigma = torch.full((self.dim,), self.init_std, device=self.device)
 
        for i in range(self.n_iters):
            # Sample candidates: (n_samples, dim)
            noise = torch.randn(self.n_samples, self.dim, device=self.device)
            z = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
 
            # Evaluate
            costs = cost_fn(z)  # (n_samples,)
 
            # Select elite
            elite_idx = torch.topk(costs, self.n_elite, largest=False).indices
            elite = z[elite_idx]  # (n_elite, dim)
 
            # Refit with momentum
            new_mu = elite.mean(dim=0)
            new_sigma = elite.std(dim=0).clamp(min=self.min_std)
 
            mu = self.momentum * mu + (1 - self.momentum) * new_mu
            sigma = self.momentum * sigma + (1 - self.momentum) * new_sigma
 
        return mu, mu, sigma
    
@torch.no_grad()
def diffusion_sample_with_noise(
    model,
    cond: torch.Tensor,
    noise_sequence: torch.Tensor,
) -> torch.Tensor:
    """
    Run the DDPM reverse chain but using *externally supplied* noise
    instead of sampling fresh noise at each step.

    Parameters
    ----------
    model          : ConditionalDiffusionPredictor
    cond           : (B, T, D) — conditioning embeddings
    noise_sequence : (B, num_steps, T, D) — one noise tensor per reverse step.
                    noise_sequence[:, s] is used at reverse step s
                    (s=0 corresponds to t=num_steps-1, s=-1 to t=1).

    Returns
    -------
    x : (B, T, D) — denoised sample
    """
    x = noise_sequence[:, 0]  # initial noise (replaces torch.randn_like)
    B = cond.shape[0]

    for idx, step in enumerate(reversed(range(model.num_steps))):
        t = torch.full((B,), step, device=cond.device, dtype=torch.long)
        x0_pred = model.denoise(x, t, cond).clamp(-10, 10)

        if step == 0:
            x = x0_pred
        else:
            ab = model._gather(model.alphas_cumprod, t, x.ndim)
            ab_prev = model._gather(model.alphas_cumprod_prev, t, x.ndim)
            alpha = model._gather(model.alphas, t, x.ndim)
            beta = model._gather(model.betas, t, x.ndim)

            mean = (
                (ab_prev.sqrt() * beta) / (1 - ab) * x0_pred
                + (alpha.sqrt() * (1 - ab_prev)) / (1 - ab) * x
            )
            var = model._gather(model.posterior_variance, t, x.ndim)

            # Use the externally provided noise instead of torch.randn_like
            # idx+1 because idx=0 was used as the initial noise
            z = noise_sequence[:, idx + 1] if (idx + 1) < noise_sequence.shape[1] else torch.zeros_like(x)
            x = mean + var.sqrt() * z

    return x