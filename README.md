# Market Latent Geometry

**Learning and analyzing the geometric structure of financial market manifolds using β-VAE and Riemannian geometry.**

This repository implements a novel framework for discovering the intrinsic geometry of financial time series through variational autoencoders. By treating the VAE decoder as a parameterization of an embedded manifold, we compute Riemannian metric tensors and geodesic distances that respect the learned curvature of market states.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv geometry
source geometry/bin/activate  # On Windows: geometry\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
cd src/

# Step 1: Train the β-VAE with specialized loss
python train_vae.py

# Step 2: Evaluate latent space quality 
python evaluate_vae.py ../results/vae_training_YYYYMMDD_HHMMSS

# Step 3: Analyze manifold geometry and clustering
python analyze_latent_geometry.py ../results/vae_training_YYYYMMDD_HHMMSS
```

## 📋 Pipeline Overview

### 🧠 `train_vae.py`
**Purpose**: Train a β-Variational Autoencoder with stability enhancements for financial data

**Key Features**:
- Specialized loss function to overcome posterior collapse
- KL capacity scheduling to prevent over-regularization  
- Orthogonality penalty for disentangled latent factors
- Robust training pipeline for financial time series

**Usage**:
```bash
python train_vae.py
```

**Outputs**:
- `../results/vae_training_YYYYMMDD_HHMMSS/` folder containing:
  - `model_weights.pth` - Trained β-VAE weights
  - `train_latent.npy`, `val_latent.npy`, `test_latent.npy` - Latent representations
  - `training_config.json` - Model configuration and hyperparameters
  - `training_history.csv` - Training metrics per epoch
  - Training diagnostics and visualizations

---

### 📊 `evaluate_vae.py`
**Purpose**: Validate latent space quality and training stability

**What it analyzes**:
- Latent space orthogonality (decorrelation success)
- Marginal normality of latent dimensions
- Training convergence and loss curves
- Adherence to VAE prior assumptions

**Usage**:
```bash
python evaluate_vae.py ../results/vae_training_YYYYMMDD_HHMMSS
```

**Outputs**:
- `correlation_analysis.png` - Latent correlation matrix and diagnostics
- `distribution_analysis.png` - Marginal distributions vs N(0,1) 
- `training_curves.png` - Loss convergence analysis
- Statistical validation reports

---

### 🌐 `analyze_latent_geometry.py`
**Purpose**: Core geometric analysis - compute manifold metrics and geodesic distances

**What it does**:
- Computes decoder Jacobians and Riemannian metric tensors G(z)
- Calculates geodesic distances that respect manifold curvature
- Compares clustering performance: Euclidean vs Geodesic vs PCA
- Provides evidence for intrinsic market manifold curvature
- Creates interactive 3D visualizations of learned geometry

**Usage**:
```bash
# Basic analysis (1000 samples, 5 clusters)
python analyze_latent_geometry.py ../results/vae_training_YYYYMMDD_HHMMSS

# Custom parameters
python analyze_latent_geometry.py ../results/vae_training_YYYYMMDD_HHMMSS \
    --n_samples 1500 \
    --n_clusters 8 \
    --output_dir custom_analysis
```

**Key Outputs**:

**🎯 Geometric Analysis**:
- `jacobians.npy` - Decoder Jacobian matrices ∂g(z)/∂z
- `riemannian_metrics.npy` - Metric tensors G(z) = J^T J
- `geodesic_distances.npy` & `euclidean_distances.npy` - Distance matrices
- `geodesic_vs_euclidean.png` - Evidence of manifold curvature

**📊 Clustering Validation**:
- `clustering_comparison.png` - Statistical comparison across methods
- `clustering_comparison.json` - Silhouette, Calinski-Harabasz, Davies-Bouldin scores
- `2d_cluster_boundaries_comparison.png` - Decision surface visualization
- `euclidean_clusters.npy` & `geodesic_clusters.npy` - Cluster labels

**🎨 Interactive 3D Visualizations** (requires plotly):
- `3d_latent_flow.html` - Time-colored trajectory through latent manifold
- `3d_clustering_comparison.html` - Euclidean vs geodesic clustering comparison
- `3d_curvature_map.html` - Local curvature visualization
- `3d_jacobian_norm_map.html` - Jacobian norm distribution

## 📁 Directory Structure

```
market-latent-geometry/
├── src/
│   ├── train_vae.py              # β-VAE training with specialized loss
│   ├── evaluate_vae.py           # Latent space validation  
│   ├── analyze_latent_geometry.py # Geometric analysis and clustering
│   ├── model.py                  # β-VAE architecture definition
│   └── config.py                 # Hyperparameter configuration
├── processed_data_simple/
│   └── latest/                   # Preprocessed S&P 500 market data
├── results/
│   └── vae_training_*/           # Training results (timestamped)
│       ├── evaluation/           # Latent space validation
│       └── geometry_analysis/    # Geometric analysis outputs
└── requirements.txt
```

## ⚙️ Configuration

Core hyperparameters in `src/config.py`:

```python
# β-VAE Architecture  
LATENT_DIM = 12        # Latent manifold dimension
HIDDEN_DIM = 128       # Hidden layer size
INPUT_DIM = 1006       # Market data dimension (503 stocks × 2 features)

# Specialized Loss Components
BETA = 1.0             # KL divergence weight 
C_CAPACITY = 4.0       # KL capacity target
LAMBDA_ORTHO = 1e-4    # Orthogonality penalty weight

# Training Parameters
LEARNING_RATE = 1e-3   # Adam learning rate
BATCH_SIZE = 32        # Batch size
NUM_EPOCHS = 80        # Maximum epochs  
PATIENCE = 200         # Early stopping patience
```

## 🎯 Research Workflow

1. **Train β-VAE**: `train_vae.py` - Learn stable latent manifold representation
2. **Validate Latents**: `evaluate_vae.py` - Ensure orthogonality and normality  
3. **Geometric Analysis**: `analyze_latent_geometry.py` - Compute manifold metrics and validate clustering
4. **Explore Results**: Open interactive HTML files and review clustering comparisons

## 📊 Key Research Findings

### Manifold Geometry Discovery:
- **Curvature Evidence**: Non-linear relationship between geodesic and Euclidean distances
- **Metric Tensor Computation**: Local geometry captured via decoder Jacobians  
- **Riemannian Structure**: Meaningful geometric structure in learned latent space

### Clustering Performance Validation:
- **Geodesic Advantage**: Improved Silhouette scores (0.07 → 0.48)
- **Calinski-Harabasz**: Better cluster separation (64 → 1,817)  
- **Davies-Bouldin**: Reduced cluster overlap (2.57 → 0.60)
- **Temporal Coherence**: Geodesic clusters show better chronological ordering

### Technical Contributions:
- **Stable Training**: Overcomes posterior collapse in financial VAEs
- **Orthogonal Latents**: Decorrelated factors via specialized loss design
- **Geometric Pipeline**: End-to-end framework for manifold learning on time series

## 🛠️ Dependencies

**Core Requirements**:
- `torch` - PyTorch for β-VAE implementation
- `numpy`, `pandas` - Data manipulation and linear algebra
- `matplotlib`, `seaborn` - Statistical plotting and analysis
- `scikit-learn` - Clustering algorithms and manifold learning
- `scipy` - Riemannian geometry computations

**Optional (for interactive 3D visualizations)**:
- `plotly` - Interactive manifold exploration

```bash
pip install plotly  # Recommended for full experience
```

## 💡 Usage Tips

- **First Run**: Use default parameters to familiarize with geometric outputs
- **Performance**: Use `--n_samples 500` for faster analysis on large datasets  
- **Clustering**: Adjust `--n_clusters` based on desired granularity
- **Visualization**: Install plotly for interactive 3D manifold exploration
- **Reproducibility**: Each run gets timestamped results for comparison

## 📖 Paper Reference

This implementation supports the research described in:

**"Market Manifolds: β-VAE Learning and Geometry on Time Series"**

*Abstract*: Financial markets exhibit complex, non-linear dynamics that traditional Euclidean models often fail to capture. This paper introduces a novel framework for learning and analyzing the underlying geometry of financial market states using β-VAEs with Riemannian metric computation.

## 🔬 Future Research Directions

- **Generative Modeling**: Sample realistic market scenarios along geodesic paths
- **Reinforcement Learning**: Train agents directly on the learned manifold  
- **Risk Management**: Use curvature as an early warning signal for instability
- **Multi-Asset**: Extend framework to FX, commodities, and crypto markets

---

*This framework establishes a new foundation for geometry-aware quantitative finance through learned manifold representations.*