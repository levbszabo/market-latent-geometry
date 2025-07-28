# Market Latent Geometry

Explore latent market factors using Autoencoders, Unsupervised Learning and Geometric methods. 

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

# Step 1: Train the VAE
python train_vae.py

# Step 2: Evaluate the trained model
python evaluate_vae.py ../results/vae_training_YYYYMMDD_HHMMSS

# Step 3: Analyze latent geometry
python analyze_latent_geometry.py ../results/vae_training_YYYYMMDD_HHMMSS

# Step 4: Perform regime modeling and financial analysis
python regime_modeling.py
```

## 📋 Scripts Overview

### 🧠 `train_vae.py`
**Purpose**: Train a Variational Autoencoder on financial market data

**What it does**:
- Loads preprocessed market data from `../processed_data_simple/latest/`
- Trains a VAE with configurable hyperparameters
- Saves model weights, latent representations, and training plots
- Creates timestamped results folder with all outputs

**Usage**:
```bash
python train_vae.py
```

**Outputs**:
- `../results/vae_training_YYYYMMDD_HHMMSS/` folder containing:
  - `model_weights.pth` - Trained model weights
  - `model_complete.pth` - Complete model object
  - `train_latent.npy`, `val_latent.npy`, `test_latent.npy` - Latent representations
  - `training_config.json` - Model configuration and hyperparameters
  - `training_history.csv` - Training metrics per epoch
  - Various plots and visualizations
  - `SUMMARY.md` - Complete training report

---

### 📊 `evaluate_vae.py`
**Purpose**: Basic evaluation of the trained VAE model

**What it does**:
- Analyzes latent space correlation (orthogonality)
- Plots training/validation loss curves
- Compares latent distributions to N(0,1)
- Generates simple evaluation report

**Usage**:
```bash
python evaluate_vae.py ../results/vae_training_YYYYMMDD_HHMMSS
```

**Outputs**:
- `../results/vae_training_YYYYMMDD_HHMMSS/evaluation/` folder containing:
  - `correlation_analysis.png` - Correlation heatmap and distribution
  - `training_curves.png` - Training and validation loss curves
  - `distribution_analysis.png` - All latent dimensions vs N(0,1)
  - `EVALUATION_SUMMARY.md` - Simple evaluation report
  - JSON and CSV files with detailed metrics

---

### 🌐 `analyze_latent_geometry.py`
**Purpose**: Part 1 of research pipeline - geometric analysis of latent manifold

**What it does**:
- Computes decoder Jacobians and Riemannian metric tensors
- Approximates geodesic vs Euclidean distances
- Performs clustering in both geometric spaces
- Creates 3D interactive visualizations of manifold structure

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

**Options**:
- `--n_samples`: Number of samples to analyze (default: 1000)
- `--n_clusters`: Number of clusters for K-means (default: 5)
- `--output_dir`: Custom output directory (default: results_path/geometry_analysis)

**Outputs**:
- `../results/vae_training_YYYYMMDD_HHMMSS/geometry_analysis/` folder containing:

**Interactive 3D Visualizations** (if plotly installed):
- `3d_latent_flow.html` - Time-colored 3D latent flow
- `3d_curvature_map.html` - Curvature-colored manifold map
- `3d_clustering_comparison.html` - Euclidean vs geodesic clustering
- `3d_jacobian_norm_map.html` - Jacobian norm visualization

**Analysis Data**:
- `jacobians.npy` - Decoder Jacobian matrices
- `riemannian_metrics.npy` - Metric tensors G(z)
- `geodesic_distances.npy` & `euclidean_distances.npy` - Distance matrices
- `euclidean_clusters.npy` & `geodesic_clusters.npy` - Cluster labels

**Summary Files**:
- `latent_geometry_summary.csv` - Complete data in tabular format
- `geometry_analysis_stats.json` - Summary statistics
- `clustering_comparison.png` - Statistical comparison of clustering approaches
- `clustering_comparison.json` - Detailed clustering quality metrics
- Various 2D analysis plots

---

### 💼 `regime_modeling.py`
**Purpose**: Part 2 of research pipeline - map geometric clustering to financial regimes

**What it does**:
- Aligns geometry analysis results with original time series data including sectors
- Computes regime-level financial metrics (returns, volatility, drawdowns, Sharpe ratios)
- Analyzes sector distributions and concentrations within each regime
- Performs regime transition analysis and persistence modeling
- Compares Euclidean vs Geodesic clustering from a financial perspective
- Validates geometric hypothesis with statistical tests

**Usage**:
```bash
# Automatic detection of latest results
python regime_modeling.py

# Specify paths manually
python regime_modeling.py \
    --results_path ../results/vae_training_YYYYMMDD_HHMMSS \
    --data_path ../processed_data_simple/latest
```

**Options**:
- `--results_path`: Path to VAE training results (auto-detects latest if not specified)
- `--data_path`: Path to processed data directory (defaults to ../processed_data_simple/latest)

**Outputs**:
- `../results/regime_analysis_YYYYMMDD_HHMMSS/` folder containing:

**Core Analysis Files**:
- `regime_stats_euclidean.csv` & `regime_stats_geodesic.csv` - Detailed regime statistics
- `financial_metrics_euclidean.csv` & `financial_metrics_geodesic.csv` - Risk-return metrics
- `sector_distribution_euclidean.csv` & `sector_distribution_geodesic.csv` - Sector compositions
- `transition_matrix_euclidean.csv` & `transition_matrix_geodesic.csv` - Regime transition probabilities
- `aligned_regime_data.parquet` - Full aligned dataset for further analysis

**Comparison Results**:
- `clustering_comparison_summary.json` - Quantitative comparison of clustering approaches
- `REGIME_ANALYSIS_SUMMARY.md` - Comprehensive executive summary

**Static Visualizations**:
- `regime_performance_comparison.png` - Risk-return analysis by regime
- `sector_distribution_heatmap.png` - Sector concentration heatmaps
- `regime_timeline.png` - Regime evolution over time
- `transition_matrices.png` - Regime transition probability matrices
- `clustering_comparison.png` - Financial coherence comparison

**Interactive Visualizations**:
- `interactive_regime_performance.html` - Interactive risk-return scatter plots
- `interactive_sector_analysis.html` - Interactive sector distribution charts
- `interactive_regime_timeline.html` - Interactive time series regime visualization

## 📁 Directory Structure

```
market-latent-geometry/
├── src/
│   ├── train_vae.py              # Main VAE training script
│   ├── evaluate_vae.py           # Basic model evaluation
│   ├── analyze_latent_geometry.py # Geometric analysis (Part 1)
│   ├── regime_modeling.py        # Regime analysis (Part 2)
│   ├── model.py                  # VAE model definition
│   └── config.py                 # Hyperparameter configuration
├── processed_data_simple/
│   └── latest/                   # Preprocessed market data
├── data/
│   └── universe/
│       └── sp500_tickers.csv     # Sector mapping for tickers
├── results/
│   ├── vae_training_*/           # Training results (timestamped)
│   │   ├── evaluation/           # Basic evaluation outputs
│   │   └── geometry_analysis/    # Geometric analysis outputs
│   └── regime_analysis_*/        # Regime analysis results (timestamped)
└── requirements.txt
```

## ⚙️ Configuration

Modify hyperparameters in `src/config.py`:

```python
# Model architecture
LATENT_DIM = 12        # Number of latent factors
HIDDEN_DIM = 128       # Hidden layer size
DROPOUT_RATE = 0.1     # Dropout rate

# Training parameters
LEARNING_RATE = 1e-4   # Learning rate
BATCH_SIZE = 32        # Batch size
NUM_EPOCHS = 100       # Maximum epochs
PATIENCE = 10          # Early stopping patience

# Loss weights
BETA = 0.0             # KL divergence weight (0 = autoencoder mode)
LAMBDA_ORTHO = 1e-3    # Orthogonality penalty weight
```

## 🎯 Typical Workflow

1. **Train the model**: Run `train_vae.py` to get a timestamped results folder
2. **Basic evaluation**: Run `evaluate_vae.py` to check training quality and orthogonality
3. **Geometric analysis**: Run `analyze_latent_geometry.py` for deep manifold analysis
4. **Regime modeling**: Run `regime_modeling.py` to map clusters to financial regimes
5. **Explore results**: Open the interactive HTML files to explore the 3D latent space and regime dynamics

## 📊 Key Insights You'll Get

### From Evaluation:
- **Orthogonality**: How well-separated your latent dimensions are
- **Training Quality**: Loss curves and convergence analysis
- **Distribution Match**: How close latent variables are to N(0,1)

### From Geometric Analysis:
- **Manifold Curvature**: Where the latent space is curved vs flat
- **Geodesic Structure**: How geometric distances differ from Euclidean
- **Temporal Flow**: How market states move through the latent space
- **Clustering Comparison**: Statistical test showing whether Euclidean or geodesic clustering is better
- **Clustering Quality**: Silhouette scores, Calinski-Harabasz index, and Davies-Bouldin index for both approaches

### From Regime Modeling:
- **Financial Regime Structure**: Risk-return profiles and characteristics of each market regime
- **Sector Dynamics**: How different sectors behave within each regime
- **Regime Transitions**: Probability matrices and persistence patterns of regime changes
- **Clustering Validation**: Financial coherence comparison between Euclidean and geodesic approaches
- **Investment Insights**: Actionable regime-based analysis for portfolio management and risk assessment

## 🛠️ Dependencies

**Core Requirements**:
- `torch` - PyTorch for deep learning
- `numpy`, `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Basic plotting
- `scikit-learn` - Clustering and manifold learning
- `scipy` - Statistical functions

**Optional (for 3D interactive plots)**:
- `plotly` - Interactive 3D visualizations

Install plotly for best experience:
```bash
pip install plotly
```

## 💡 Tips

- **First run**: Start with default parameters to get familiar with outputs
- **Large datasets**: Use `--n_samples 500` for faster geometric analysis
- **Custom analysis**: Adjust `--n_clusters` based on your data structure
- **Interactive exploration**: Install plotly and open HTML files in browser for 3D interaction
- **Results comparison**: Each training run gets a unique timestamp for easy comparison
- **Regime analysis**: The script automatically detects the latest VAE results, but you can specify paths manually for specific experiments
- **Financial validation**: Pay attention to the clustering comparison results to understand which approach provides better financial interpretability
- **Sector insights**: Use the sector distribution analysis to understand how market regimes relate to sector rotation patterns

---

*This pipeline provides a complete framework for understanding the geometric structure of financial market data through learned latent representations.*