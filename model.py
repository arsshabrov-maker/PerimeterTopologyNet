import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def map_perimeter_to_2d(t, shape='triangle', n_teeth=8):
  """
    Maps perimeter coordinate t (mod 1.0) onto 2D coordinates (x, y)
    of a closed contour: 'triangle', 'circle', 'square', or 'gear'.
    """
  t = torch.remainder(t, 1.0)
  angle = 2.0 * np.pi * t

  if shape == 'circle':
    return torch.cos(angle), torch.sin(angle)

  elif shape == 'square':
    s = t * 4.0
    x = torch.zeros_like(t)
    y = torch.zeros_like(t)

    m0 = s < 1.0
    m1 = (s >= 1.0) & (s < 2.0)
    m2 = (s >= 2.0) & (s < 3.0)
    m3 = s >= 3.0

    x = torch.where(m0, -1.0 + 2.0 * s, x)
    y = torch.where(m0, -1.0, y)
    x = torch.where(m1, 1.0, x)
    y = torch.where(m1, -1.0 + 2.0 * (s - 1.0), y)
    x = torch.where(m2, 1.0 - 2.0 * (s - 2.0), x)
    y = torch.where(m2, 1.0, y)
    x = torch.where(m3, -1.0, x)
    y = torch.where(m3, 1.0 - 2.0 * (s - 3.0), y)
    return x, y

  elif shape == 'triangle':
    v0x, v0y = 0.0, 1.0
    v1x, v1y = -0.8660254, -0.5
    v2x, v2y = 0.8660254, -0.5

    s = t * 3.0
    m0 = s < 1.0
    m1 = (s >= 1.0) & (s < 2.0)

    u0 = s
    u1 = s - 1.0
    u2 = s - 2.0

    x = torch.where(
        m0,
        v0x + (v1x - v0x) * u0,
        torch.where(m1, v1x + (v2x - v1x) * u1, v2x + (v0x - v2x) * u2),
    )
    y = torch.where(
        m0,
        v0y + (v1y - v0y) * u0,
        torch.where(m1, v1y + (v2y - v1y) * u1, v2y + (v0y - v2y) * u2),
    )
    return x, y

  elif shape == 'gear':
    tooth_wave = torch.tanh(3.5 * torch.sin(n_teeth * angle))
    radius = 0.82 + 0.22 * tooth_wave
    return radius * torch.cos(angle), radius * torch.sin(angle)


class TriangleTopologyLayer(nn.Module):
  """
    Evolutionary Champion Basis (`PerimeterTopologyNet`):
      - Contour: Equilateral Triangle (3 sharp vertices inducing piecewise-linear folds)
      - Anchors: 2 equidistant perimeter anchors with affine displacement
      - Extracted Invariants: chord length, cyclic perimeter order, and radial distances
    """

  def __init__(self, in_dim, out_dim, act_type='gelu'):
    super().__init__()
    self.n_anchors = 2
    self.act_type = act_type

    anchors = torch.tensor([0.0, 0.5]).view(1, 1, 2)
    self.register_buffer('anchors', anchors)

    self.proj = nn.Linear(in_dim, out_dim * 2, bias=True)
    # 4 geometric invariants per neuron: [chord_length, cyclic_order, radius_1, radius_2]
    self.geom_combiner = nn.Parameter(torch.randn(1, out_dim, 4) * 0.3)
    self.geom_bias = nn.Parameter(torch.zeros(1, out_dim))

  def forward(self, x):
    B = x.size(0)
    delta = self.proj(x).view(B, -1, 2)
    t_pts = self.anchors + 0.35 * torch.tanh(delta)

    px, py = map_perimeter_to_2d(t_pts, shape='triangle')

    # 1. Chord length between the two moving perimeter points
    chord = torch.sqrt(
        (px[:, :, 0] - px[:, :, 1]) ** 2
        + (py[:, :, 0] - py[:, :, 1]) ** 2
        + 1e-6
    )

    # 2. Cyclic perimeter order
    t_mod = torch.remainder(t_pts, 1.0)
    order = torch.sin(2.0 * np.pi * (t_mod[:, :, 1] - t_mod[:, :, 0]))

    # 3. Radial distances from contour center (captures corner vs. edge-midpoint position)
    rad0 = torch.sqrt(px[:, :, 0] ** 2 + py[:, :, 0] ** 2 + 1e-6)
    rad1 = torch.sqrt(px[:, :, 1] ** 2 + py[:, :, 1] ** 2 + 1e-6)

    geom_feats = torch.stack([chord, order, rad0, rad1], dim=-1)
    raw_out = (geom_feats * self.geom_combiner).sum(dim=-1) + self.geom_bias

    if self.act_type == 'tanh':
      return torch.tanh(raw_out)
    elif self.act_type == 'sin':
      return torch.sin(raw_out)
    return F.gelu(raw_out)


class PerimeterTopologyNet(nn.Module):
  """Compact neural network powered by the evolutionary TriangleTopologyLayer."""

  def __init__(self, in_dim, out_dim, hidden_dim=12, act_type='gelu'):
    super().__init__()
    self.layer1 = TriangleTopologyLayer(in_dim, hidden_dim, act_type=act_type)
    self.head = nn.Linear(hidden_dim, out_dim)

  def forward(self, x):
    return self.head(self.layer1(x))
