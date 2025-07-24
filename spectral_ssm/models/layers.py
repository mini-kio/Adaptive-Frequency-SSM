"""
Spectral-Latent SSM Layers based on S6 Architecture
Implements frequency domain state compression with S6 improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from einops import rearrange, repeat

from ..utils.fft_utils import (
    slice_low_frequencies, 
    pad_and_reconstruct, 
    AdaptiveFrequencyMask,
    frequency_dropout,
    spectral_norm_regularization,
    CirculantMatrix
)


class SpectralS6Block(nn.Module):
    """
    S6-based Spectral-Latent SSM Block with frequency domain compression
    
    Key improvements over standard S6:
    1. Frequency domain state compression
    2. Adaptive frequency selection
    3. Multi-scale processing
    4. Spectral regularization
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        compression_ratio: float = 0.5,
        use_adaptive_mask: bool = True,
        use_spectral_norm: bool = True,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.compression_ratio = compression_ratio
        self.k = int(d_state * compression_ratio)  # Compressed state dimension
        self.use_adaptive_mask = use_adaptive_mask
        self.use_spectral_norm = use_spectral_norm
        
        # Input projection following S6
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # Convolution layer for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        
        # S6 activation
        self.activation = "silu"
        self.act = nn.SiLU()
        
        # SSM parameters - modified for spectral domain
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
            
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # A parameter - frequency domain version
        if use_spectral_norm:
            # Use circulant matrix for better frequency domain properties
            self.A_log = nn.Parameter(torch.randn(self.d_inner, self.k, **factory_kwargs))
        else:
            self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state, **factory_kwargs))
        
        # S6 structured matrix initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        
        # Frequency domain components
        if use_adaptive_mask:
            self.freq_mask = AdaptiveFrequencyMask(d_state, compression_ratio)
        else:
            self.freq_mask = None
            
        # Spectral gating for high-frequency information
        self.spectral_gate = nn.Linear(self.d_inner, self.d_inner, bias=bias, **factory_kwargs)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(self.d_inner, **factory_kwargs)
        
    def forward(self, hidden_states, inference_params=None):
        """
        Forward pass with spectral state compression
        
        Args:
            hidden_states: (B, L, D)
            inference_params: For compatibility with S6
        """
        batch, seqlen, dim = hidden_states.shape
        
        # Input projection
        xz = self.in_proj(hidden_states)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Convolution for local context
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :seqlen]  # Causal convolution
        x = rearrange(x, "b d l -> b l d")
        
        # Activation
        x = self.act(x)
        
        # SSM computation with spectral compression
        y = self.spectral_ssm(x)
        
        # Gating mechanism
        z = self.act(z)
        y = y * z
        
        # Output projection
        output = self.out_proj(y)
        return output
    
    def spectral_ssm(self, u):
        """
        Spectral SSM computation with frequency domain compression
        
        Args:
            u: Input tensor (B, L, d_inner)
        """
        batch, seqlen, d_inner = u.shape
        
        # Get SSM parameters
        delta, B, C = self.get_ssm_params(u)  # (B, L, d_inner), (B, L, d_state), (B, L, d_state)
        
        # Discretization
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))  # (B, L, d_inner, d_state)
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)  # (B, L, d_inner, d_state)
        
        # Spectral domain processing
        if self.training and hasattr(self, 'freq_mask') and self.freq_mask is not None:
            # Use adaptive frequency masking during training
            return self.spectral_ssm_adaptive(deltaA, deltaB_u, C, u)
        else:
            # Use fixed compression ratio
            return self.spectral_ssm_fixed(deltaA, deltaB_u, C, u)
    
    def spectral_ssm_adaptive(self, deltaA, deltaB_u, C, u):
        """SSM with adaptive frequency selection"""
        batch, seqlen, d_inner, d_state = deltaA.shape
        
        # Convert to frequency domain
        deltaA_freq = slice_low_frequencies(deltaA, self.k, dim=-1)
        deltaB_u_freq = slice_low_frequencies(deltaB_u, self.k, dim=-1)
        
        # Apply spectral normalization for stability
        if self.use_spectral_norm:
            deltaA_freq = spectral_norm_regularization(deltaA_freq)
        
        # Frequency dropout for regularization
        if self.training:
            deltaA_freq = frequency_dropout(deltaA_freq, p=0.1, training=True)
        
        # SSM recurrence in compressed frequency domain
        x_freq = self.parallel_scan_spectral(deltaA_freq, deltaB_u_freq)
        
        # Reconstruct to original dimension
        x = pad_and_reconstruct(x_freq, d_state, dim=-1)
        
        # Output projection
        y = torch.einsum('bldn,bln->bld', x, C)
        
        # Add skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        
        return y
    
    def spectral_ssm_fixed(self, deltaA, deltaB_u, C, u):
        """SSM with fixed compression ratio"""
        batch, seqlen, d_inner, d_state = deltaA.shape
        
        # Simple frequency domain compression
        deltaA_compressed = deltaA[..., :self.k]
        deltaB_u_compressed = deltaB_u[..., :self.k]
        C_compressed = C[..., :self.k]
        
        # SSM recurrence in compressed domain
        x_compressed = self.parallel_scan_spectral(deltaA_compressed, deltaB_u_compressed)
        
        # Output projection in compressed domain
        y = torch.einsum('bldn,bln->bld', x_compressed, C_compressed)
        
        # Add skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        
        return y
    
    def parallel_scan_spectral(self, A, B):
        """
        Parallel scan in spectral domain with improved numerical stability
        """
        # Implementation of parallel scan for spectral domain
        # This is a simplified version - full implementation would use
        # more sophisticated parallel scan algorithms
        
        batch, seqlen, d_inner, d_state = A.shape
        
        # Initialize state list to avoid in-place operations
        x_list = []
        h = torch.zeros(batch, d_inner, d_state, device=A.device, dtype=A.dtype)
        
        # Sequential scan without in-place operations
        for i in range(seqlen):
            if i == 0:
                h = B[:, i].clone()
            else:
                h = A[:, i] * h + B[:, i]
            x_list.append(h.unsqueeze(1))
        
        # Concatenate all states
        x = torch.cat(x_list, dim=1)  # (batch, seqlen, d_inner, d_state)
        
        return x
    
    def get_ssm_params(self, u):
        """Get SSM parameters following S6 design"""
        batch, seqlen, d_inner = u.shape
        
        # Project input to get dt, B, C
        x_dbl = self.x_proj(u)  # (B, L, dt_rank + 2*d_state)
        
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Process dt
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt + self.dt_proj.bias)
        
        return dt, B, C


class SpectralResidualBlock(nn.Module):
    """
    Residual block with spectral processing for multi-scale information
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        compression_ratio: float = 0.5,
        num_scales: int = 3,
        **kwargs
    ):
        super().__init__()
        
        self.spectral_block = SpectralS6Block(
            d_model=d_model,
            d_state=d_state,
            compression_ratio=compression_ratio,
            **kwargs
        )
        
        # Multi-scale processing
        self.num_scales = num_scales
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(kwargs.get('dropout', 0.0)),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(kwargs.get('dropout', 0.0))
        )
    
    def forward(self, x):
        # Pre-norm for spectral block
        x_norm = self.norm1(x)
        spectral_out = self.spectral_block(x_norm)
        x = x + spectral_out
        
        # Pre-norm for FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


class MultiHeadSpectralAttention(nn.Module):
    """
    Multi-head attention with spectral processing
    Can be used alongside spectral SSM for hybrid architectures
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_spectral: bool = True,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_spectral = use_spectral
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if use_spectral:
            self.spectral_processor = AdaptiveFrequencyMask(
                d_state=self.d_head, 
                compression_ratio=compression_ratio
            )
    
    def forward(self, x):
        batch, seqlen, d_model = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch, seqlen, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seqlen, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seqlen, self.num_heads, self.d_head).transpose(1, 2)
        
        # Spectral processing for attention scores
        if self.use_spectral and hasattr(self, 'spectral_processor'):
            q, _ = self.spectral_processor(q, dim=-1)
            k, _ = self.spectral_processor(k, dim=-1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seqlen, d_model)
        out = self.out_proj(out)
        
        return out