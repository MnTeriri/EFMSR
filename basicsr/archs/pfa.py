# reference (Progressive Focused Transformer for Single Image Super-Resolution, PFT) https://github.com/LabShuHangGU/PFT-SR
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import smm_cuda

from basicsr.archs.arch_util import trunc_normal_


class SMM_QmK(Function):
    """
    A custom PyTorch autograd Function for sparse matrix multiplication (SMM) of
    query (Q) and key (K) matrices, based on given sparse indices.

    This function leverages a CUDA-implemented kernel for efficient computation.

    Forward computation:
        Computes the sparse matrix multiplication using a custom CUDA function.

    Backward computation:
        Computes the gradients of A and B using a CUDA-implemented backward function.
    """

    @staticmethod
    def forward(ctx, A, B, index):
        """
        Forward function for Sparse Matrix Multiplication QmK.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            A: Input tensor A (Query matrix).
            B: Input tensor B (Key matrix).
            index: Index tensor specifying the sparse multiplication positions.

        Returns:
            Tensor: Result of the sparse matrix multiplication.
        """
        # Save input tensors for backward computation
        ctx.save_for_backward(A, B, index)

        # Call the custom CUDA forward function for sparse matrix multiplication
        return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Backward function for Sparse Matrix Multiplication QmK.

        Args:
            ctx: Autograd context to retrieve saved tensors.
            grad_output: Gradient of the output from the forward pass.

        Returns:
            Tuple: Gradients of the inputs A and B, with None for the index as it is not trainable.
        """
        # Retrieve saved tensors from the forward pass
        A, B, index = ctx.saved_tensors

        # Compute gradients using the custom CUDA backward function
        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )

        # Return gradients for A and B, no gradient for index
        return grad_A, grad_B, None


class SMM_AmV(Function):
    """
    A custom PyTorch autograd Function for sparse matrix multiplication (SMM)
    between an activation matrix (A) and a value matrix (V), guided by sparse indices.

    This function utilizes a CUDA-optimized implementation for efficient computation.

    Forward computation:
        Computes the sparse matrix multiplication using a custom CUDA function.

    Backward computation:
        Computes the gradients of A and B using a CUDA-implemented backward function.
    """

    @staticmethod
    def forward(ctx, A, B, index):
        """
        Forward function for Sparse Matrix Multiplication AmV.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            A: Input tensor A (Activation matrix).
            B: Input tensor B (Value matrix).
            index: Index tensor specifying the sparse multiplication positions.

        Returns:
            Tensor: Result of the sparse matrix multiplication.
        """
        # Save tensors for backward computation
        ctx.save_for_backward(A, B, index)

        # Call the custom CUDA forward function
        return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Backward function for Sparse Matrix Multiplication AmV.

        Args:
            ctx: Autograd context to retrieve saved tensors.
            grad_output: Gradient of the output from the forward pass.

        Returns:
            Tuple: Gradients of the inputs A and B, with None for the index as it is not trainable.
        """
        # Retrieve saved tensors from the forward pass
        A, B, index = ctx.saved_tensors

        # Compute gradients using the custom CUDA backward function
        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )

        # Return gradients for A and B, no gradient for index
        return grad_A, grad_B, None


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention (MSA).

    Args:
        dim (int): Number of input channels.
        layer_id (int): Index of the current layer in the network.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        num_topk (tuple[int]): Number of top-k attention values retained for sparsity.
        qkv_bias (bool, optional): If True, add a learnable bias to the query, key, and value tensors. Default: True.
    """

    def __init__(self, dim, layer_id, window_size, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-20

        # define a parameter table of relative position bias
        if dim > 100:
            # for classical SR
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        else:
            # for lightweight SR
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), 1))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0):
        r"""
        Args:
        qkvp (Tensor): Input tensor containing query, key, value tokens, and LePE positional encoding matrix with shape (num_windows * b, n, c * 4),
        pfa_values (Tensor or None): Precomputed attention values for Progressive Focusing Attention (PFA). If None, standard attention is applied.
        pfa_indices (Tensor or None): Index tensor for Progressive Focusing Attention (PFA), indicating which attention values should be retained or discarded.
        rpi (Tensor): Relative position index tensor, encoding positional information for tokens.
        mask (Tensor or None, optional): Attention mask tensor.
        shift (int, optional): Indicates whether window shifting is applied (e.g., 0 for no shift, 1 for shifted windows). Default: 0.
        """
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # Standard Attention Computation
        if pfa_indices[shift] is None:
            attn = (q @ k.transpose(-2, -1))  # b_, self.num_heads, n, n
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(
                0)  # nH, Wh*Ww, Wh*Ww
            if not self.training:  # Check if in inference mode
                attn.add_(relative_position_bias)  # only in inference
            else:
                attn = attn + relative_position_bias  # Non-inplace if training

            if shift:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
        # # Sparse Attention Computation using SMM_QmK
        else:
            topk = pfa_indices[shift].shape[-1]
            q = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            k = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads).transpose(-2, -1)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            attn = SMM_QmK.apply(q, k, smm_index).view(b_, self.num_heads, n, topk)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0).expand(b_,
                                                                                                              self.num_heads,
                                                                                                              n,
                                                                                                              n)  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices[shift])
            if not self.training:  # Check if in inference mode
                attn.add_(relative_position_bias)  # only in inference
            else:
                attn = attn + relative_position_bias  # Non-inplace if training

        # # Use in-place operations where possible (only in inference mode)
        # if not self.training:  # Check if in inference mode
        #     attn = torch.softmax(attn, dim=-1, out=attn)  # 原地softmax
        # else:
        #     attn = self.softmax(attn)  # Non-inplace if training

        attn = self.softmax(attn)  # Non-inplace if training

        # Apply Hadamard product for PFA and normalize.
        # if pfa_values[shift] is not None:
        #     if not self.training: # only in inference
        #         attn.mul_(pfa_values[shift])
        #         attn.add_(self.eps)
        #         denom = attn.sum(dim=-1, keepdim=True).add_(self.eps)
        #         attn.div_(denom)
        #     else:
        #         attn = (attn * pfa_values[shift])
        #         attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        if pfa_values[shift] is not None:
            attn = (attn * pfa_values[shift])
            attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        # If sparsification is enabled, select top-k attention values and save the corresponding indexes
        if self.topk < self.window_size[0] * self.window_size[1]:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices[shift] is not None:
                pfa_indices[shift] = torch.gather(pfa_indices[shift], dim=-1, index=topk_indices)
            else:
                pfa_indices[shift] = topk_indices

        # Save the current attention results as PFA maps.
        pfa_values[shift] = attn

        # # Save the attention map as a .npy file for visualization or further analysis
        # # Scatter the attention values back to their original indices
        # # attn_npy has shape (batch_size * num_windows, num_heads, n, n)
        # if pfa_indices[shift] is None:
        #     attn_npy = attn
        # else:
        #     attn_npy = torch.zeros((b_, self.num_heads, n, n), device=attn.device).scatter(-1, pfa_indices[shift], attn)
        # # Define the path where the attention map will be saved
        # attention_save_path = f"./results/Attention_map/PFT_light_attention_map_w32_L{self.layer_id}.npy"
        # os.makedirs("./results/Attention_map", exist_ok=True)
        # # Save the attention map only if the file does not already exist to avoid overwriting
        # if not os.path.exists(attention_save_path):
        #     np.save(attention_save_path, attn_npy.cpu().detach().numpy())

        # Check whether sparsification has been applied; if so, use SMM_AmV for computation, otherwise perform standard matrix multiplication A @ V.
        if pfa_indices[shift] is None:
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        else:
            topk = pfa_indices[shift].shape[-1]
            attn = attn.view(b_ * self.num_heads, n, topk)
            v = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            x = (SMM_AmV.apply(attn, v, smm_index).view(b_, self.num_heads, n, c // self.num_heads) + v_lepe).transpose(
                1, 2).reshape(b_, n, c)

        # only in inference. After use, delete unnecessary variables to free memory
        # if not self.training:
        #     del q, k, v, relative_position_bias
        #     torch.cuda.empty_cache()  # Clear the unused cache

        x = self.proj(x)
        return x, pfa_values, pfa_indices

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0
        if self.layer_id < 2:
            # attn = (q @ k.transpose(-2, -1))
            flops += self.num_heads * n * (self.dim // self.num_heads) * n
            #  x = (attn @ v)
            flops += self.num_heads * n * n * (self.dim // self.num_heads)
        else:
            # attn = (q @ k.transpose(-2, -1))
            flops += self.num_heads * n * (self.dim // self.num_heads) * self.num_topk[self.layer_id - 2]
            #  x = (attn @ v)
            flops += self.num_heads * n * self.num_topk[self.layer_id] * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops
