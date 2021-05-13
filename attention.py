import math
import copy
import torch
import torch.nn as nn
from global_config import DEVICE


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        ctx_dim=None,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()
        new_ctx_layer_shape = ctx_layer.size()[:-2] + (self.all_head_size,)
        ctx_layer = ctx_layer.view(*new_ctx_layer_shape)

        return ctx_layer


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers=1,
        nhead=1,
        dropout=0.1,
        dim_feedforward=128,
        max_seq_length=5000,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        self.encoder = TransformerEncoder(
            TransformerLayer(
                d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
            ),
            num_layers=num_layers,
        )
        self.decoder = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, attention_mask=None):
        seq_length = input.size()[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=DEVICE)
        positions_embedding = (
            self.pos_encoder(position_ids).unsqueeze(0).expand(input.size())
        )  # (seq_length, d_model) => (batch_size, seq_length, d_model)
        input = input + positions_embedding
        input = self.norm(input)
        hidden = self.encoder(input, attention_mask=attention_mask)
        out = self.decoder(hidden)  # (batch_size, seq_len, hidden_dim)
        # ([CLS] token embedding, full output, last hidden layer)
        out = (out[:, 0, :], out, hidden)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead=1, dim_feedforward=128, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = Attention(hidden_size, nhead, dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attention_mask=None):
        src_1 = self.self_attention(src, src, attention_mask=attention_mask)
        src = src + self.dropout1(src_1)
        src = self.norm1(src)
        src_2 = self.fc(src)
        src = src + self.dropout2(src_2)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)

    def forward(self, src, attention_mask=None):
        for layer in self.layers:
            new_src = layer(src, attention_mask=attention_mask)
            src = src + new_src
        return src


class CrossAttentionEncoder(nn.Module):
    def __init__(self, layer, n_layers):
        super(CrossAttentionEncoder, self).__init__()
        self.layers = _get_clones(layer, n_layers)

    def forward(self, src, ctx, attention_mask=None):
        for layer in self.layers:
            cross_src, cross_ctx = layer(src, ctx, attention_mask=attention_mask)
            src = src + cross_src
            ctx = ctx + cross_ctx

        return cross_src, cross_ctx


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, ctx_size, n_heads=1, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.src_cross_attention = Attention(
            hidden_size, n_heads, dropout, ctx_dim=ctx_size
        )
        self.ctx_cross_attention = Attention(
            ctx_size, n_heads, dropout, ctx_dim=hidden_size
        )
        self.src_self_attention = Attention(hidden_size, n_heads, dropout)
        self.ctx_self_attention = Attention(ctx_size, n_heads, dropout)
        self.src_fc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.ctx_fc = nn.Sequential(nn.Linear(ctx_size, ctx_size), nn.ReLU())
        self.src_norm1 = nn.LayerNorm(hidden_size)
        self.ctx_norm1 = nn.LayerNorm(ctx_size)
        self.src_norm2 = nn.LayerNorm(hidden_size)
        self.ctx_norm2 = nn.LayerNorm(ctx_size)
        self.src_dropout = nn.Dropout(dropout)
        self.ctx_dropout = nn.Dropout(dropout)

    def forward(self, src, ctx, attention_mask=None):
        src = self.src_cross_attention(src, ctx, attention_mask=attention_mask)
        ctx = self.ctx_cross_attention(ctx, src, attention_mask=attention_mask)

        cross_src = self.src_self_attention(src, src, attention_mask)
        cross_src = cross_src + self.src_dropout(cross_src)
        cross_src = self.src_norm1(cross_src)

        cross_ctx = self.ctx_self_attention(ctx, ctx, attention_mask)
        cross_ctx = cross_ctx + self.ctx_dropout(cross_ctx)
        cross_ctx = self.ctx_norm1(cross_ctx)

        cross_src = self.src_fc(src)
        cross_src = cross_src + self.src_dropout(cross_src)
        cross_src = self.src_norm2(cross_src)

        cross_ctx = self.ctx_fc(ctx)
        cross_ctx = cross_ctx + self.ctx_dropout(cross_ctx)
        cross_ctx = self.ctx_norm2(cross_ctx)

        return cross_src, cross_ctx
