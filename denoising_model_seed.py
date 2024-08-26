import numpy as np
import torch
import torch.nn as nn
from math import log as ln

d_model = 400  # Embedding Size
d_ff = 512  # FeedForward dimension
d_k = d_v = 32  # dimension of K(=Q), V
n_heads = 1  # number of heads in Multi-Head Attention


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding.unsqueeze(-1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
                Q: [batch_size, n_heads, len_q, d_k]
                K: [batch_size, n_heads, len_k, d_k]
                V: [batch_size, n_heads, len_v(=len_k), d_v]
                attn_mask: [batch_size, n_heads, seq_len, seq_len]
                '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V):
        '''
                input_Q: [batch_size, len_q, d_model]
                input_K: [batch_size, len_k, d_model]
                input_V: [batch_size, len_v(=len_k), d_model]
                attn_mask: [batch_size, seq_len, seq_len]
                '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        output = self.ln(output + residual)
        return output


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
                inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        output = self.ln((output + residual))  # [batch_size, seq_len, d_model]
        return output


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        """
                enc_inputs: [batch_size, src_len, d_model]
                enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(FiLM, self).__init__()

        self.conv_gamma = nn.Conv1d(in_channels=1, out_channels=input_dim, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv1d(in_channels=1, out_channels=input_dim, kernel_size=3, padding=1)

        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)

    def forward(self, x, condition):
        # gamma_conv = self.fc_gamma(condition)
        # beta_conv = self.fc_beta(condition)

        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)

        # 对输入特征x进行缩放和偏移，实现条件特征调整输入特征
        y = gamma * x + beta
        return y


class DualBranchDenoisingModel(nn.Module):
    def __init__(self, feats=64):
        super(DualBranchDenoisingModel, self).__init__()
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 3, padding=1),
                          Conv1d(feats, feats, 3, padding=1),
                          ),
            EncoderLayer(),
            EncoderLayer(),
            EncoderLayer(),
        ])

        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 3, padding=1),
                          Conv1d(feats, feats, 3, padding=1),
                          ),
            EncoderLayer(),
            EncoderLayer(),
            EncoderLayer(),
        ])

        self.embed = PositionalEncoding(feats)

        self.bridge = nn.ModuleList([
            FiLM(d_model, 1),
            FiLM(d_model, 1),
            FiLM(d_model, 1),
            FiLM(d_model, 1),
        ])

        self.conv_out = nn.Sequential(Conv1d(feats, feats, 3, padding=1),
                                      Conv1d(feats, 1, 3, padding=1),
                                      )

    def forward(self, x, cond, noise_scale):
        noise_embed = self.embed(noise_scale)
        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed))

        for x, layer in zip(xs, self.stream_cond):
            cond = layer(cond) + x

        return self.conv_out(cond)


if __name__ == '__main__':
    net = DualBranchDenoisingModel(80)
    # leaf = frontend.Leaf(sample_rate=400, n_filters=128, window_len=65, window_stride=40).cuda()
    x = torch.randn(10, 1, 512)
    y = torch.randn(10, 1)
    z = net(x, x, y)

    print(z.shape)

# summary(net, ((x,x,y)))
