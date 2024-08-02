import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, in_channels, dropout_rate):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head
        self.h = n_head
        self.layer_norm = nn.LayerNorm(in_channels)
        self.linear_q = nn.Linear(in_channels, in_channels)
        self.linear_k = nn.Linear(in_channels, in_channels)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.linear_out = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, q, k, v, mask=None, pos_k=None):
        n_batch = q.size(0)
        seq_len = q.size(1)  # 获取序列长度

        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        q = self.linear_q(q).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(v).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if pos_k is not None:
            pos_k = pos_k.unsqueeze(0).unsqueeze(0).expand(n_batch, self.h, -1, -1)
            scores = scores + pos_k[:, :, :scores.size(-2), :scores.size(-1)]

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0).expand(n_batch, self.h, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v).transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.dropout(self.linear_out(x)), attn


class StreamingTransformer(nn.Module):
    def __init__(self, n_head, in_channels, num_layers, dropout_rate, history_size):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttention(n_head, in_channels, dropout_rate)
            for _ in range(num_layers)
        ])
        self.history_size = history_size
        self.history_k = [None] * len(self.layers)
        self.history_v = [None] * len(self.layers)
        self.pos_enc = self.compute_positional_encodings(1000)  # 假设最大长度为1000

    def compute_positional_encodings(self, length):
        pos_enc = torch.zeros(length, length)
        for i in range(length):
            for j in range(length):
                pos_enc[i, j] = i - j
        return pos_enc

    def forward(self, x):
        n_batch, seq_len, _ = x.size()
        outputs = []

        for i, layer in enumerate(self.layers):
            q = x
            if self.history_k[i] is not None and self.history_v[i] is not None:
                k = torch.cat([self.history_k[i], x], dim=1)
                v = torch.cat([self.history_v[i], x], dim=1)
            else:
                k, v = x, x

            pos_k = self.pos_enc[:k.size(1), :k.size(1)].to(x.device)
            mask = torch.ones(seq_len, k.size(1)).to(x.device)
            out, attn = layer(q, k, v, mask, pos_k)

            # 确保历史长度正确
            if self.history_k[i] is None:
                self.history_k[i] = k
                self.history_v[i] = v
            else:
                self.history_k[i] = k[:, -self.history_size:] if k.size(1) > self.history_size else k
                self.history_v[i] = v[:, -self.history_size:] if v.size(1) > self.history_size else v

            outputs.append(out)

        return outputs[-1]


def process_streaming_data(model, chunk_size, input_stream):

    for chunk in input_stream:
        output = model(chunk)
        yield output[:, -chunk_size:, :]


# Example usage:
n_head = 8
in_channels = 512
num_layers = 6
dropout_rate = 0.1
chunk_size = 80
history_size = 400

model = StreamingTransformer(n_head, in_channels, num_layers, dropout_rate, history_size)

# Simulate an input stream of audio data
input_stream = (torch.randn(1, chunk_size, in_channels) for _ in range(10))

# Process the input stream
output_stream = process_streaming_data(model, chunk_size, input_stream)

# Print the shape of the outputs
for i, output in enumerate(output_stream):
    print(f"Output chunk {i}: {output.shape}")
