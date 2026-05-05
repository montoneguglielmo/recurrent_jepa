import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )
        self.output_size = emb_dim

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )
        self.output_size = output_dim or input_dim

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class CNNNet(nn.Module):
    def __init__(self, num_conv_layers=2, num_filters=None, input_size=32, output_size=None):
        super().__init__()

        if num_filters is None:
            num_filters = [6 * (2 ** i) for i in range(num_conv_layers)]
        assert len(num_filters) == num_conv_layers

        # Build conv layers dynamically
        in_channels = 3
        self.conv_layers = nn.ModuleList()
        for out_channels in num_filters:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5))
            in_channels = out_channels

        self.pool = nn.MaxPool2d(2, 2)

        # Dummy forward pass to compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            for conv in self.conv_layers:
                dummy = self.pool(F.relu(conv(dummy)))
                print('dummy shape', dummy.shape)
            flat_size = dummy.numel()

        fc1_size = flat_size // 8
        fc2_size = flat_size // 16
        if output_size is None:
            output_size = flat_size // 32
        self.fc1 = nn.Linear(flat_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class TimeWrapper(nn.Module):
    """Wraps any network to handle video inputs (B, T, ...)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.output_size = backbone.output_size

    def forward(self, x):
        B, T, *rest = x.shape
        x = x.reshape(B * T, *rest)

        x = self.backbone(x)

        x = x.reshape(B, T, *x.shape[1:])
        return x


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding (DDPM-style)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long → (B, dim) float"""
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaLNMLPBlock(nn.Module):
    """Residual MLP block with AdaLN-zero conditioning. Reuses modulate()."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x, c: (B, T, D)"""
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        return x + gate * self.mlp(modulate(self.norm(x), shift, scale))


class ConditionalDiffusionPredictor(nn.Module):
    """
    DDPM-style conditional diffusion predictor for JEPA latent space.

    Drop-in replacement for TimeWrapper(MLP(...)): accepts (B, T, D),
    returns (B, T, D). No TimeWrapper needed.

    Training (.training=True):
        Samples random t, corrupts x via q_sample, returns denoised x0.
        The caller's MSE loss (pred - target).pow(2) IS the DDPM
        x0-prediction objective — no changes to jepa.py required.

    Eval (.training=False):
        Runs full DDPM reverse sampling conditioned on x.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_steps: int = 100,
        num_layers: int = 4,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.output_size = embed_dim

        # Timestep embedding: sinusoidal → two-layer MLP
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Denoiser
        self.layers = nn.ModuleList([
            AdaLNMLPBlock(embed_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Noise schedule buffers
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer("posterior_variance", posterior_variance)

    def _gather(self, buf: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Gather schedule values at t and reshape for broadcasting over (B, T, D)."""
        vals = buf.gather(0, t)                      # (B,)
        return vals.reshape(-1, *([1] * (ndim - 1))) # (B, 1, ..., 1)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0). x0: (B,T,D), t: (B,)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self._gather(self.sqrt_alphas_cumprod, t, x0.ndim)
        sm = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim)
        return sa * x0 + sm * noise

    def denoise(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Predict x0 from noisy x_t and condition. x_t, cond: (B,T,D), t: (B,)."""
        # Conditioning: input embedding + timestep embedding (broadcast over T)
        c = cond + self.time_embed(t).unsqueeze(1)  # (B, 1, D) broadcasts to (B, T, D)
        h = x_t
        for layer in self.layers:
            h = layer(h, c)
        h = self.output_norm(h)
        return x_t + self.output_proj(h)            # residual: near-identity at init

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        """Full DDPM reverse chain starting from noise, conditioned on cond."""
        x = torch.randn_like(cond)
        B = cond.shape[0]

        for step in reversed(range(self.num_steps)):
            t = torch.full((B,), step, device=cond.device, dtype=torch.long)
            x0_pred = self.denoise(x, t, cond).clamp(-10, 10)

            if step == 0:
                x = x0_pred
            else:
                ab     = self._gather(self.alphas_cumprod,      t, x.ndim)
                ab_prev = self._gather(self.alphas_cumprod_prev, t, x.ndim)
                alpha  = self._gather(self.alphas,               t, x.ndim)
                beta   = self._gather(self.betas,                t, x.ndim)
                mean = (
                    (ab_prev.sqrt() * beta) / (1 - ab) * x0_pred
                    + (alpha.sqrt() * (1 - ab_prev)) / (1 - ab) * x
                )
                var = self._gather(self.posterior_variance, t, x.ndim)
                x = mean + var.sqrt() * torch.randn_like(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) embeddings from state encoder.
        Returns (B, T, D) predicted embeddings.
        """
        if self.training:
            B = x.shape[0]
            t = torch.randint(0, self.num_steps, (B,), device=x.device)
            x_t = self.q_sample(x, t)
            return self.denoise(x_t, t, cond=x)
        else:
            return self.sample(cond=x)