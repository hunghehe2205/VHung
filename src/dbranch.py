import torch
from torch import nn


class DBranch(nn.Module):
    """Dense per-frame anomaly head.

    Input  : visual features post-GCN, shape (B, T, in_dim).
    Output : score map s_t in [0, 1], shape (B, T).

    Padded frames (per ``padding_mask``) are zeroed in the output so they
    do not contribute to losses or downstream metrics.
    """

    def __init__(self, in_dim: int = 512, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden // 2, 1, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden // 2)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x.transpose(1, 2)                          # (B, in_dim, T)
        x = self.conv1(x).transpose(1, 2)              # (B, T, hidden)
        x = self.dropout(self.act(self.ln1(x)))
        x = x.transpose(1, 2)                          # (B, hidden, T)
        x = self.conv2(x).transpose(1, 2)              # (B, T, hidden/2)
        x = self.dropout(self.act(self.ln2(x)))
        x = x.transpose(1, 2)                          # (B, hidden/2, T)
        x = self.conv3(x).transpose(1, 2).squeeze(-1)  # (B, T)
        s = torch.sigmoid(x)

        if padding_mask is not None:
            keep = (~padding_mask.bool()).to(s.dtype)
            s = s * keep
        return s
