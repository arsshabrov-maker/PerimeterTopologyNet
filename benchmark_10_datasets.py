import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PerimeterTopologyNet


def get_10_datasets():
  raw_list = []
  d = load_breast_cancer()
  raw_list.append(('1. Breast Cancer', d.data, d.target, 'cls', 2))
  d = load_wine()
  raw_list.append(('2. Wine Chemistry', d.data, d.target, 'cls', 3))
  d = load_iris()
  raw_list.append(('3. Iris', d.data, d.target, 'cls', 3))
  d = load_digits()
  raw_list.append(('4. Digits 8x8', d.data, d.target, 'cls', 10))
  d = load_diabetes()
  raw_list.append(('5. Diabetes Reg', d.data, d.target, 'reg', 1))

  X, y = make_circles(n_samples=600, noise=0.12, factor=0.45, random_state=42)
  raw_list.append(('6. Nested Circles', X, y, 'cls', 2))
  X, y = make_moons(n_samples=600, noise=0.18, random_state=42)
  raw_list.append(('7. Two Moons', X, y, 'cls', 2))

  rng = np.random.RandomState(42)
  X = rng.uniform(-3.0, 3.0, size=(600, 6))
  y = (np.sin(4.0 * X[:, 0]) + np.cos(3.0 * X[:, 1] * X[:, 2]) > 0).astype(int)
  raw_list.append(('8. Gear Waves', X, y, 'cls', 2))

  X = rng.randn(600, 8)
  y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
  outlier_mask = rng.rand(600) < 0.30
  X[outlier_mask] += rng.randn(outlier_mask.sum(), 8) * 6.0
  raw_list.append(('9. Noisy XOR 30%', X, y, 'cls', 2))

  X = rng.uniform(-2.0, 2.0, size=(600, 10))
  y = np.sin(X[:, 0] + X[:, 1]) * X[:, 2] + 0.5 * (X[:, 3] ** 2 - X[:, 4] ** 2)
  raw_list.append(('10. Formula Reg', X, y, 'reg', 1))

  prepared = []
  for name, X_raw, y_raw, task, n_out in raw_list:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=0.25, random_state=42
    )
    sc = StandardScaler()
    X_tr = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
    X_te = torch.tensor(sc.transform(X_te), dtype=torch.float32)
    if task == 'cls':
      y_tr_t = torch.tensor(y_tr, dtype=torch.long)
      y_te_t = torch.tensor(y_te, dtype=torch.long)
    else:
      ym, ys = y_tr.mean(), y_tr.std() + 1e-6
      y_tr_t = torch.tensor((y_tr - ym) / ys, dtype=torch.float32).unsqueeze(1)
      y_te_t = torch.tensor((y_te - ym) / ys, dtype=torch.float32).unsqueeze(1)
    prepared.append((name, X_tr, y_tr_t, X_te, y_te_t, task, n_out))
  return prepared


def main():
  datasets = get_10_datasets()
  print('=' * 82)
  print('PERIMETER TOPOLOGY NET (TRIANGLE) vs PARAMETER-MATCHED MLP')
  print('=' * 82)

  tri_scores, mlp_scores = [], []
  for name, X_tr, y_tr, X_te, y_te, task, n_out in datasets:
    in_dim = X_tr.size(1)

    # 1. TriangleTopologyNet
    torch.manual_seed(42)
    tri_net = PerimeterTopologyNet(in_dim, n_out, hidden_dim=12)
    tri_p = sum(p.numel() for p in tri_net.parameters())
    opt = torch.optim.AdamW(tri_net.parameters(), lr=0.015, weight_decay=1e-3)
    for _ in range(90):
      tri_net.train()
      opt.zero_grad()
      out = tri_net(X_tr)
      loss = (
          F.cross_entropy(out, y_tr)
          if task == 'cls'
          else F.mse_loss(out, y_tr)
      )
      loss.backward()
      opt.step()

    tri_net.eval()
    with torch.no_grad():
      pred = tri_net(X_te)
      sc_t = (
          (pred.argmax(dim=1) == y_te).float().mean().item()
          if task == 'cls'
          else max(0.0, 1.0 - F.mse_loss(pred, y_te).item() / y_te.var().item())
      )
    tri_scores.append(sc_t)

    # 2. Parameter-Matched Standard MLP
    torch.manual_seed(42)
    h_mlp = max(6, int(tri_p // (in_dim + n_out + 1)))
    mlp = nn.Sequential(
        nn.Linear(in_dim, h_mlp), nn.GELU(), nn.Linear(h_mlp, n_out)
    )
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=0.015, weight_decay=1e-3)
    for _ in range(90):
      mlp.train()
      opt_m.zero_grad()
      out = mlp(X_tr)
      loss = (
          F.cross_entropy(out, y_tr)
          if task == 'cls'
          else F.mse_loss(out, y_tr)
      )
      loss.backward()
      opt_m.step()

    mlp.eval()
    with torch.no_grad():
      pred = mlp(X_te)
      sc_m = (
          (pred.argmax(dim=1) == y_te).float().mean().item()
          if task == 'cls'
          else max(0.0, 1.0 - F.mse_loss(pred, y_te).item() / y_te.var().item())
      )
    mlp_scores.append(sc_m)

    print(
        f'{name:<22} | Triangle ({tri_p:<4}p): {sc_t*100:6.2f}%  vs  MLP:'
        f' {sc_m*100:6.2f}%'
    )

  print('=' * 82)
  print(
      f'MEAN ACROSS 10 TASKS   | Triangle: {np.mean(tri_scores)*100:6.2f}%  vs '
      f' MLP: {np.mean(mlp_scores)*100:6.2f}%'
  )
  print('=' * 82)


if __name__ == '__main__':
  main()
