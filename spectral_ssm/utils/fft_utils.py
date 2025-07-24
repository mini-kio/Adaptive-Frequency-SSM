"""
FFT Utilities for Spectral-Latent SSM
Implements frequency domain operations for state compression
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def real_fft(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Real FFT with proper normalization for spectral analysis"""
    return torch.fft.rfft(x, dim=dim, norm='ortho')


def real_ifft(x: torch.Tensor, n: Optional[int] = None, dim: int = -1) -> torch.Tensor:
    """Inverse real FFT with proper normalization"""
    return torch.fft.irfft(x, n=n, dim=dim, norm='ortho')


def dct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Discrete Cosine Transform using FFT implementation"""
    N = x.size(dim)
    x_pad = torch.cat([x, x.flip(dims=[dim])], dim=dim)
    
    X = torch.fft.fft(x_pad, dim=dim)
    X = X.narrow(dim, 0, N)
    
    # DCT-II scaling
    k = torch.arange(N, dtype=x.dtype, device=x.device)
    W = torch.exp(-1j * torch.pi * k / (2 * N))
    
    if dim == -1:
        X = X * W
    else:
        shape = [1] * x.ndim
        shape[dim] = N
        W = W.view(shape)
        X = X * W
    
    return X.real


def idct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse Discrete Cosine Transform"""
    N = x.size(dim)
    
    # Create frequency scaling
    k = torch.arange(N, dtype=x.dtype, device=x.device)
    W = torch.exp(1j * torch.pi * k / (2 * N))
    
    if dim == -1:
        X = x * W
    else:
        shape = [1] * x.ndim
        shape[dim] = N
        W = W.view(shape)
        X = x * W
    
    # Extend for IFFT
    X_ext = torch.zeros(*x.shape[:-1], 2*N, dtype=torch.complex64, device=x.device)
    X_ext.narrow(dim, 0, N).copy_(X.to(torch.complex64))
    
    # Conjugate symmetry for real result
    if dim == -1:
        X_ext[..., N:] = X_ext[..., 1:N].flip(dims=[-1]).conj()
    else:
        X_ext = X_ext.transpose(dim, -1)
        X_ext[..., N:] = X_ext[..., 1:N].flip(dims=[-1]).conj()
        X_ext = X_ext.transpose(dim, -1)
    
    result = torch.fft.ifft(X_ext, dim=dim).real
    return result.narrow(dim, 0, N) * 2


def get_top_k_frequencies(x: torch.Tensor, k: int, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-k frequency components based on magnitude
    Returns: (top_k_values, indices)
    """
    freqs = real_fft(x, dim=dim)
    magnitudes = torch.abs(freqs)
    
    # Get top-k indices
    _, indices = torch.topk(magnitudes, k, dim=dim)
    
    # Gather top-k values
    top_k_freqs = torch.gather(freqs, dim, indices)
    
    return top_k_freqs, indices


def slice_low_frequencies(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    Slice the first k low-frequency components after FFT
    This is the main compression technique
    """
    freqs = real_fft(x, dim=dim)
    return freqs.narrow(dim, 0, k)


def pad_and_reconstruct(x_compressed: torch.Tensor, original_size: int, dim: int = -1) -> torch.Tensor:
    """
    Pad compressed frequencies and reconstruct signal
    """
    # Create padded tensor
    pad_size = original_size // 2 + 1 - x_compressed.size(dim)  # rfft output size
    
    if pad_size > 0:
        # Create padding tensor
        pad_shape = list(x_compressed.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=x_compressed.dtype, device=x_compressed.device)
        
        # Concatenate
        x_padded = torch.cat([x_compressed, padding], dim=dim)
    else:
        x_padded = x_compressed
    
    return real_ifft(x_padded, n=original_size, dim=dim)


class AdaptiveFrequencyMask(torch.nn.Module):
    """
    Learnable mask for adaptive frequency selection
    """
    def __init__(self, d_state: int, compression_ratio: float = 0.5):
        super().__init__()
        self.d_state = d_state
        self.k = int(d_state * compression_ratio)
        
        # Learnable importance weights for each frequency
        freq_size = d_state // 2 + 1  # rfft output size
        self.freq_weights = torch.nn.Parameter(torch.ones(freq_size))
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adaptive frequency masking
        Returns: (compressed_freqs, selection_mask)
        """
        freqs = real_fft(x, dim=dim)
        
        # Compute importance scores
        importance = torch.sigmoid(self.freq_weights / self.temperature.abs())
        
        # Get top-k frequencies based on learned importance
        _, indices = torch.topk(importance, self.k)
        indices = indices.sort().values  # Keep frequency order
        
        # Create selection mask
        mask = torch.zeros_like(importance, dtype=torch.bool)
        mask[indices] = True
        
        # Apply mask
        if dim == -1:
            compressed_freqs = freqs[..., mask]
        else:
            mask_expanded = mask.view([1] * dim + [mask.size(0)] + [1] * (freqs.ndim - dim - 1))
            compressed_freqs = freqs[mask_expanded.expand_as(freqs)].view(*freqs.shape[:dim], self.k, *freqs.shape[dim+1:])
        
        return compressed_freqs, mask


def frequency_dropout(x: torch.Tensor, p: float = 0.1, training: bool = True) -> torch.Tensor:
    """
    Randomly drop frequency components during training for regularization
    """
    if not training or p == 0:
        return x
    
    mask = torch.rand_like(x.real) > p
    return x * mask.to(x.dtype)


def multi_scale_fft(x: torch.Tensor, scales: list = [1, 2, 4]) -> list:
    """
    Multi-scale FFT analysis for different temporal resolutions
    """
    results = []
    for scale in scales:
        if scale == 1:
            results.append(real_fft(x))
        else:
            # Downsample and analyze
            downsampled = F.avg_pool1d(x.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)
            results.append(real_fft(downsampled))
    
    return results


class CirculantMatrix(torch.nn.Module):
    """
    Circulant matrix for frequency domain operations
    A circulant matrix can be diagonalized by FFT
    """
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.first_row = torch.nn.Parameter(torch.randn(size))
    
    def get_eigenvalues(self) -> torch.Tensor:
        """Get eigenvalues via FFT of first row"""
        return torch.fft.fft(self.first_row)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply circulant matrix multiplication via FFT"""
        x_freq = torch.fft.fft(x, dim=-1)
        eigenvals = self.get_eigenvalues()
        result_freq = x_freq * eigenvals
        return torch.fft.ifft(result_freq, dim=-1).real


def spectral_norm_regularization(A_freq: torch.Tensor, max_eigenval: float = 1.0) -> torch.Tensor:
    """
    Spectral normalization for frequency domain matrices
    Ensures stability by constraining largest eigenvalue
    """
    eigenvals = torch.abs(A_freq)
    max_eig = torch.max(eigenvals)
    
    if max_eig > max_eigenval:
        return A_freq * (max_eigenval / max_eig)
    return A_freq