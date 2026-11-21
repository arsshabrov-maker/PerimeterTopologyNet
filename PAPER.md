# PerimeterTopologyNet: Evolutionary Discovery of Compact Geometric Neuron Bases on Closed Contours

**Author:** Arsenii Shabrov  
**Date:** November 2026  

## Abstract
Standard artificial neurons compute unbounded scalar dot products followed by pointwise activations. We propose **PerimeterTopologyNet**, a geometric neural basis that maps input-driven displacements onto closed 2D contours (**Circle**, **Triangle**, **Square**, and **Toothed Gear**) and extracts topological invariants—including chord lengths, cyclic point ordering, chord intersections, Euler regions, and radial distances from the contour center. Using an evolutionary search with a parameter-count complexity penalty across **10 diverse classification and regression benchmarks**, evolution converges on a compact **Equilateral Triangle basis** (`2` equidistant anchors, `490` average parameters). Across parameter-matched evaluations, `TriangleTopologyNet` outperforms standard MLPs on **6 out of 10 benchmarks**, notably improving accuracy on **Noisy XOR with 30% extreme outliers** (`75.33%` vs. `70.67%`) and **8×8 Handwritten Digits** (`95.33%` vs. `92.67%`).

## 1. Perimeter-Topology Formulation
Given $K$ equidistant anchor positions $\mathbf{a} \in [0, 1)^K$ on a closed perimeter, an input vector $\mathbf{x} \in \mathbb{R}^d$ induces bounded cyclic displacements:
$$\mathbf{t} = \big(\mathbf{a} + 0.35 \tanh(W\mathbf{x} + \mathbf{b})\big) \bmod 1.0$$
The perimeter coordinates $\mathbf{t}$ are projected onto 2D coordinates $(p_x, p_y)$ along one of four closed contours:
* **Circle:** Smooth periodic boundary.
* **Equilateral Triangle:** Three $120^\circ$ vertices inducing sharp piecewise-linear directional folds.
* **Square:** Four orthogonal $90^\circ$ edges.
* **Toothed Gear:** Periodic radial teeth modulating radius with angular frequency.

From the resulting 2D point configuration, each neuron extracts geometric invariants (pairwise chord lengths, cyclic ordering $\sin(2\pi(t_j - t_i))$, soft chord crossings, Euler zones, and radial distances $r_i = \sqrt{p_{x,i}^2 + p_{y,i}^2}$) and combines them into a scalar activation.

## 2. Evolutionary Tournament Results (10 Benchmarks)
Across 4 generations (48 evaluated genomes) on 10 real and synthetic tasks (*Breast Cancer, Wine, Iris, Digits, Diabetes, Nested Circles, Two Moons, Gear Waves, Noisy XOR + 30% Outliers, Multi-Factor Regression*):
* **Triangle Contour:** Max `84.63%` | Mean `81.89%` (24 variants)
* **Square Contour:** Max `83.87%` | Mean `82.27%` (7 variants)
* **Circle Contour:** Max `83.99%` | Mean `82.01%` (15 variants)
* **Gear Contour:** Max `79.40%` | Mean `76.62%` (2 variants)

The winning compact genome (**Triangle**, `2` anchors, `affine` shift, invariants: `chords + order + radii`, `490` avg. params) outperformed a parameter-matched MLP (`481` avg. params) on **6 of 10 datasets**, demonstrating strong natural outlier boundedness (`+4.66%` on 30% corrupted XOR).
