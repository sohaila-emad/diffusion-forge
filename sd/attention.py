import torch 
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.d_embed = d_embed
        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        #x: (batch_size, seq_len, d_embed)
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interm_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        #(batch_size, seq_len, Dim) -> (batch_size, seq_len, n_heads, d_head/n_heads)-> (batch_size, n_heads, seq_len, d_head)
        q = q.view(interm_shape).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_head)
        k = k.view(interm_shape).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_head)
        v = v.view(interm_shape).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_head)

        weight = q @ k.transpose(-2, -1) * (self.d_head ** -0.5)  # (batch_size, n_heads, seq_len, seq_len)

        if causal_mask:
            #create a lower triangular matrix of shape (seq_len, seq_len) to mask out future positions
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
            weight = weight.masked_fill(mask == 0, float('-inf'))

        weight = F.softmax(weight, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)

        output = weight @ v  # (batch_size, n_heads, seq_len, d_head/n_heads)
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)  # (batch_size, seq_len, d_embed)

        return self.out_proj(output)


class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.d_embed = d_embed
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed , bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed , bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        #x: (batch_size, seq_len_x, d_embed)
        #context: (batch_size, seq_len_context, Dim_kv)

        input_shape = x.shape
        batch_size, seq_len_x, d_embed = input_shape

        interm_shape_x = (batch_size, -1, self.n_heads, self.d_head)
        interm_shape_context = (batch_size, context.size(1), self.n_heads, self.d_head)

        q = self.q_proj(x).view(interm_shape_x).transpose(1, 2)  # (batch_size, n_heads, seq_len_x, d_head)
        k = self.k_proj(context).view(interm_shape_context).transpose(1, 2)  # (batch_size, n_heads, seq_len_context, d_head)
        v = self.v_proj(context).view(interm_shape_context).transpose(1, 2)  # (batch_size, n_heads, seq_len_context, d_head)

        weight = q @ k.transpose(-2, -1) * (self.d_head ** -0.5)  # (batch_size, n_heads, seq_len_x, seq_len_context)
        weight = F.softmax(weight, dim=-1)  # (batch_size, n_heads, seq_len_x, seq_len_context)

        #here we do not need to apply any mask as we want the picture to attend to all the text tokens, not just the previous ones.
        
        output = weight @ v  # (batch_size, n_heads, seq_len_x, d_head/n_heads)
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)  # (batch_size, seq_len_x, d_embed)

        return self.out_proj(output)