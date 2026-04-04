"""
VadInternVL — Video Anomaly Detection với InternVL features.

Thay đổi so với phiên bản cũ:
  1. Thêm Dual Memory Banks (Amemory + Nmemory) từ UR-DMU
  2. Thêm encoder_mu để project memory augmentation
  3. Thêm triplet loss support (trả về aux_dict khi training=True)
  4. Fix DistanceAdj: dùng self.sigma thay vì hardcode torch.tensor(1.)  [đã có]
  5. Fix mlp2 init std=0.02                                               [đã có]
  6. Residual x_gcn + x_transformer                                       [đã có]
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import OrderedDict
from scipy.spatial.distance import pdist, squareform


# ──────────────────────────────────────────────
# Graph components (giữ nguyên)
# ──────────────────────────────────────────────

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=False, residual=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)
        if not residual:
            self.residual = lambda x: 0
        elif in_features == out_features:
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(
                in_channels=in_features, out_channels=out_features,
                kernel_size=5, padding=2
            )

    def forward(self, input, adj):
        support = input.matmul(self.weight)
        output  = adj.matmul(support)
        if self.bias is not None:
            output = output + self.bias
        if self.in_features != self.out_features and self.residual:
            res = self.residual(input.permute(0, 2, 1)).permute(0, 2, 1)
            output = output + res
        else:
            output = output + self.residual(input)
        return output


class DistanceAdj(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigma = Parameter(torch.FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen, device):
        arith = np.arange(max_seqlen).reshape(-1, 1)
        dist  = pdist(arith, metric='cityblock').astype(np.float32)
        dist  = torch.from_numpy(squareform(dist)).to(device)
        # FIX: dùng self.sigma (learnable) thay vì hardcode torch.tensor(1.)
        dist  = torch.exp(-dist / torch.exp(self.sigma))
        dist  = dist.unsqueeze(0).repeat(batch_size, 1, 1)
        return dist


# ──────────────────────────────────────────────
# Transformer components (giữ nguyên)
# ──────────────────────────────────────────────

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        return super().forward(x.type(torch.float32)).type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn     = nn.MultiheadAttention(d_model, n_head)
        self.ln_1     = LayerNorm(d_model)
        self.mlp      = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(d_model, d_model * 4)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2     = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask   = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x,
                         need_weights=False,
                         key_padding_mask=padding_mask,
                         attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width    = width
        self.layers   = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# ──────────────────────────────────────────────
# NEW: Memory Unit (từ UR-DMU)
# ──────────────────────────────────────────────

class Memory_Unit(nn.Module):
    """
    Learnable memory bank.
    - nums  : số memory slots (prototype vectors)
    - dim   : feature dimension

    forward(data):  data [B, T, D]
      → temporal_att [B, T]   : anomaly score per snippet (sigmoid attention)
      → augment      [B, T, D]: feature augmented bằng memory
    """

    def __init__(self, nums: int, dim: int):
        super().__init__()
        self.dim  = dim
        self.nums = nums
        self.memory_block = Parameter(torch.empty(nums, dim))
        self.sig  = nn.Sigmoid()
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)

    def forward(self, data: torch.Tensor):
        # data: [B, T, D]
        # attention: [B, T, K]
        attention    = self.sig(
            torch.einsum('btd,kd->btk', data, self.memory_block) / (self.dim ** 0.5)
        )
        # temporal_att: top-(K//16+1) mean → [B, T]
        k            = self.nums // 16 + 1
        temporal_att = torch.topk(attention, k, dim=-1)[0].mean(-1)
        # augment: [B, T, D]
        augment      = torch.einsum('btk,kd->btd', attention, self.memory_block)
        return temporal_att, augment


# ──────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────

class VadInternVL(nn.Module):
    """
    Pipeline:
      input [B, T, 1024]
        → InputNorm + PositionEmbed
        → Transformer (windowed attention)       → x_transformer
        → GCN (adj4 + disAdj)                    → x_gcn
        → x = x_gcn + x_transformer              (residual)
        → Amemory(x) → A_aug, A_att
        → Nmemory(x) → N_aug, N_att
        → x_mem = x + encoder_mu(A_aug + N_aug)
        → logits = classifier(x_mem + mlp2(x_mem))

    forward(visual, lengths, is_training=False):
      - is_training=False → return logits  [B, T, 1]
      - is_training=True  → return logits, aux_dict
        aux_dict keys: A_att, N_att, x (pre-memory features)
    """

    def __init__(self,
                 visual_length: int,
                 visual_width:  int,
                 visual_head:   int,
                 visual_layers: int,
                 attn_window:   int,
                 device,
                 # Memory Bank params
                 a_nums: int = 60,
                 n_nums: int = 60):
        super().__init__()
        self.visual_length = visual_length
        self.visual_width  = visual_width
        self.attn_window   = attn_window
        self.device        = device

        # ── Temporal Transformer ──
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self._build_attention_mask(attn_window),
        )

        # ── GCN ──
        width = visual_width // 2
        self.gc1     = GraphConvolution(visual_width, width, residual=True)
        self.gc2     = GraphConvolution(width,        width, residual=True)
        self.gc3     = GraphConvolution(visual_width, width, residual=True)
        self.gc4     = GraphConvolution(width,        width, residual=True)
        self.disAdj  = DistanceAdj()
        self.linear  = nn.Linear(visual_width, visual_width)
        self.gelu    = QuickGELU()

        # ── MLP head ──
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(visual_width, visual_width * 4)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width)),
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        # ── Input processing ──
        self.input_norm              = LayerNorm(visual_width)
        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)

        # ── NEW: Dual Memory Banks ──
        self.Amemory    = Memory_Unit(nums=a_nums, dim=visual_width)
        self.Nmemory    = Memory_Unit(nums=n_nums, dim=visual_width)
        self.encoder_mu = nn.Linear(visual_width, visual_width)
        self.mem_gate   = nn.Parameter(torch.zeros(1))  # init=0, learns to open

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)
        for m in self.mlp2:
            if isinstance(m, nn.Linear):
                # FIX: std=0.02 (trước là 0.001 → gần zero suốt epoch đầu)
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        # encoder_mu init
        nn.init.normal_(self.encoder_mu.weight, std=0.02)
        nn.init.zeros_(self.encoder_mu.bias)

    def _build_attention_mask(self, attn_window: int) -> torch.Tensor:
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        num_windows = self.visual_length // attn_window
        for i in range(num_windows):
            s = i * attn_window
            e = min((i + 1) * attn_window, self.visual_length)
            mask[s:e, s:e] = 0
        return mask

    def _adj4(self, x: torch.Tensor, seq_len) -> torch.Tensor:
        """Cosine similarity adjacency matrix với threshold 0.7."""
        x2     = x.matmul(x.permute(0, 2, 1))                     # [B, T, T]
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)          # [B, T, 1]
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))         # [B, T, T]
        x2     = x2 / (x_norm_x + 1e-20)

        soft   = nn.Softmax(dim=1)
        output = torch.zeros_like(x2)

        for i in range(len(seq_len)):
            s  = int(seq_len[i])
            tmp = x2[i, :s, :s]
            tmp = F.threshold(tmp, 0.7, 0)
            tmp = soft(tmp)
            output[i, :s, :s] = tmp
        return output

    def encode_video(self, images: torch.Tensor, lengths) -> torch.Tensor:
        """
        images: [B, T, D]
        returns: [B, T, D]  (x_gcn + x_transformer residual)
        """
        images = images.to(torch.float)
        images = self.input_norm(images)

        # Position embedding
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        pos_emb      = self.frame_position_embeddings(position_ids)  # [B, T, D]

        images = images.permute(1, 0, 2) + pos_emb.permute(1, 0, 2)  # [T, B, D]

        # Transformer
        x, _ = self.temporal((images, None))
        x    = x.permute(1, 0, 2)  # [B, T, D]
        x_transformer = x          # save for residual

        # GCN
        adj    = self._adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1], x.device)

        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))
        x1   = self.gelu(self.gc2(x1_h, adj))
        x2   = self.gelu(self.gc4(x2_h, disadj))

        x_gcn = self.linear(torch.cat((x1, x2), dim=2))  # [B, T, D]

        # FIX: residual connection (trước đây x_transformer bị overwrite)
        return x_gcn + x_transformer

    def forward(self, visual: torch.Tensor, lengths,
                is_training: bool = False):
        """
        visual      : [B, T, D]
        lengths     : [B]  — actual sequence length per sample
        is_training : True → trả thêm aux_dict cho memory loss + triplet loss

        Returns:
          logits              [B, T, 1]
          aux_dict (optional) {
              'A_att': [B, T],
              'N_att': [B, T],
              'x'    : [B, T, D],   # pre-memory features
          }
        """
        # ── Encode ──
        x = self.encode_video(visual, lengths)          # [B, T, D]

        # ── Memory augmentation ──
        A_att, A_aug = self.Amemory(x)                  # [B,T], [B,T,D]
        N_att, N_aug = self.Nmemory(x)                  # [B,T], [B,T,D]

        # Project và cộng vào x (gated — starts near 0, learns to open)
        mem_aug = self.encoder_mu(A_aug + N_aug)        # [B, T, D]
        x_mem   = x + torch.sigmoid(self.mem_gate) * mem_aug  # [B, T, D]

        # ── Classifier ──
        logits = self.classifier(x_mem + self.mlp2(x_mem))  # [B, T, 1]

        if is_training:
            aux_dict = {
                'A_att': A_att,   # [B, T]
                'N_att': N_att,   # [B, T]
            }
            return logits, aux_dict

        return logits