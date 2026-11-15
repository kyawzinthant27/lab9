import math

import torch
from torch import nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout_rate):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        scores = query @ key.transpose(-2, -1)
        scores = scores / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        output = self.dropout(attention) @ value

        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(
            math.sqrt(self.d_k),
            dropout_rate,
        )

    def split_heads(self, x):
        batch = x.size(0)
        x = x.view(batch, -1, self.n_heads, self.d_k)

        return x.transpose(1, 2)

    def group_heads(self, x):
        batch = x.size(0)
        x = x.transpose(1, 2).contiguous()

        return x.view(batch, -1, self.n_heads * self.d_k)

    def forward(self, query, key, value, mask=None):
        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))

        if mask is not None:
            mask = mask.unsqueeze(1)

        x, attention = self.attention(query, key, value, mask)
        x = self.W_o(self.group_heads(x))

        return x, attention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(F.relu(self.w_1(x)))

        return self.w_2(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).float().unsqueeze(1)

        scale = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)

        self.register_buffer("pe", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()

        self.attn_layer = MultiHeadAttention(
            d_model, n_heads, dropout_rate
        )
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.ff_layer = PositionwiseFeedForward(
            d_model, d_ff, dropout_rate
        )
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attention, _ = self.attn_layer(x, x, x, mask)
        x = self.attn_layer_norm(x + self.dropout(attention))

        output = self.ff_layer(x)
        x = self.ff_layer_norm(x + self.dropout(output))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        pad_idx,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.tok_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_idx
        )
        self.pos_embedding = PositionalEncoding(
            d_model, dropout_rate
        )

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        x = self.pos_embedding(self.tok_embedding(x))

        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()

        self.attn_layer = MultiHeadAttention(
            d_model, n_heads, dropout_rate
        )
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.enc_attn_layer = MultiHeadAttention(
            d_model, n_heads, dropout_rate
        )
        self.enc_attn_layer_norm = nn.LayerNorm(
            d_model, eps=1e-6
        )

        self.ff_layer = PositionwiseFeedForward(
            d_model, d_ff, dropout_rate
        )
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, memory, src_mask, tgt_mask):
        output, _ = self.attn_layer(x, x, x, tgt_mask)
        x = self.attn_layer_norm(x + self.dropout(output))

        output, attention = self.enc_attn_layer(
            x, memory, memory, src_mask
        )
        x = self.enc_attn_layer_norm(x + self.dropout(output))

        output = self.ff_layer(x)
        x = self.ff_layer_norm(x + self.dropout(output))

        return x, attention


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        pad_idx,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.tok_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_idx
        )
        self.pos_embedding = PositionalEncoding(
            d_model, dropout_rate
        )

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.pos_embedding(self.tok_embedding(x))

        for layer in self.layers:
            x, attention = layer(x, memory, src_mask, tgt_mask)

        return self.layer_norm(x), attention


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, generator, pad_idx):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.pad_idx = pad_idx

    def get_pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1)

    def get_subsequent_mask(self, x):
        length = x.size(1)

        return torch.ones(
            1, length, length,
            dtype=torch.bool,
            device=x.device,
        ).tril()

    def forward(self, source, target):
        src_mask = self.get_pad_mask(source)

        tgt_mask = (
            self.get_pad_mask(target)
            & self.get_subsequent_mask(target)
        )

        memory = self.encoder(source, src_mask)

        output, attention = self.decoder(
            target, memory, src_mask, tgt_mask
        )

        return self.generator(output), attention


class NoamOptim:
    def __init__(self, optimizer, d_model, factor, warmup):
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.warmup = warmup
        self.steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.steps += 1

        rate = self.factor * self.d_model ** -0.5 * min(
            self.steps ** -0.5,
            self.steps * self.warmup ** -1.5,
        )

        for group in self.optimizer.param_groups:
            group["lr"] = rate

        self.optimizer.step()
