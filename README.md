# PerimeterTopologyNet: Evolutionary Geometric Neuron Bases on Closed Contours

**PerimeterTopologyNet** replaces standard unbounded dot-product neurons with **closed-contour geometric units**. Input signals displace movable nodes along the perimeter of a closed shape (**Triangle**, **Circle**, **Square**, or **Toothed Gear**), and each neuron computes topological invariants—**chord lengths**, **cyclic ordering**, **chord intersections**, and **radial distances**—to produce bounded, geometry-aware activations.

---

## 🧬 Evolutionary Contour Tournament (10 Datasets)

To discover the most compact and generalizable geometric basis without manual guesswork, we ran an evolutionary search with a parameter-count penalty across **10 diverse real and synthetic benchmarks** (*Breast Cancer, Wine, Iris, Digits, Diabetes, Nested Circles, Two Moons, Periodic Gear Waves, Noisy XOR with 30% Outliers, and Multi-Factor Formula Regression*):

| Contour Geometry | Evaluated Genomes | Mean 10-Task Score | Best 10-Task Score |
| :--- | :---: | :---: | :---: |
| **Equilateral Triangle** | 24 | 81.89% | **84.63%** |
| **Circle** | 15 | 82.01% | 83.99% |
| **Square** | 7 | **82.27%** | 83.87% |
| **Toothed Gear** | 2 | 76.62% | 79.40% |

Evolution converged on a lightweight **2-Anchor Equilateral Triangle basis** (`490` avg. parameters) utilizing **chord length**, **cyclic order**, and **radial distances** (which naturally encode whether a moving point sits near a sharp triangle vertex or an edge midpoint).

---

## 📊 Head-to-Head Benchmark: Evolutionary Triangle Champion vs. Matched MLP

Evaluated at equal parameter budgets (`~490` vs `~481` parameters) across all 10 tasks:

| Benchmark Task | Metric | Triangle Champion (Ours) | Matched Standard MLP |
| :--- | :---: | :---: | :---: |
| **1. Breast Cancer** | Accuracy | 96.50% | **98.60%** |
| **2. Wine Chemistry** | Accuracy | **100.00%** | 97.78% |
| **3. Iris** | Accuracy | **100.00%** | **100.00%** |
| **4. Digits 8×8** | Accuracy | **95.33%** | 92.67% |
| **5. Diabetes Regression** | $R^2$ | 36.49% | **49.40%** |
| **6. Nested Circles** | Accuracy | **99.33%** | 98.67% |
| **7. Two Moons** | Accuracy | **98.00%** | 96.67% |
| **8. Gear Waves** | Accuracy | 53.33% | **54.00%** |
| **9. Noisy XOR (30% Extreme Outliers)** | Accuracy | **75.33%** | 70.67% |
| **10. Multi-Factor Formula** | $R^2$ | **91.97%** | 91.63% |

### Key Takeaways
* **6 Wins, 1 Tie, 3 Losses** against a parameter-matched MLP across 10 varied domains.
* **Natural Outlier Robustness (`+4.66%` on 30% Outliers):** Because displacements wrap around a closed perimeter, extreme input corruptions cannot explode activation magnitudes.
* **Sharp Geometric Folding:** The equilateral triangle's three $120^\circ$ corners provide crisp piecewise-linear transitions with minimal parameters.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python benchmark_10_datasets.py
```
