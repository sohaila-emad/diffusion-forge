import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(1, n_tokens, n_embed))

    def forward(self, tokens: torch.Tensor) -> torch.FloatTensor:
        #tokens: (batch_size, seq_len)
        token_embeddings = self.token_embedding(tokens)  # (batch_size, seq_len, embed_dim)
        position_embeddings = self.position_embedding[:, :tokens.size(1), :]  # (1, seq_len, embed_dim)

        return token_embeddings + position_embeddings  # (batch_size, seq_len, embed_dim)


class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed) 
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, n_embed * 4)
        self.linear_2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        # x: (batch_size, seq_len, embed_dim)
        residual = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)  # (batch_size, seq_len, embed_dim)
        x = x + residual

        #feedforward network
        residual = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.linear_2(x)
        x = x + residual

        return x
    

class CLIP(nn.Module):
    def __init__(self):
        super().__init__() 
        self.embedding = CLIPEmbedding(49408, 768, 77) # vocab_size, embed_dim, max_length

        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12) # 12 transformer layers
        ])

        self.layernorm = nn.LayerNorm(768) # final layer normalization

    def forward(self, tokens: torch.Tensor) -> torch.FloatTensor:
        #tokens: (batch_size, seq_len)
        tokens = tokens.type(torch.long) # ensure tokens are of type long for embedding lookup

        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        state = self.embedding(tokens) # (batch_size, seq_len, embed_dim)

        for layer in self.layers:
            state = layer(state) # (batch_size, seq_len, embed_dim)

        #(batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        output = self.layernorm(state) 

        return output


