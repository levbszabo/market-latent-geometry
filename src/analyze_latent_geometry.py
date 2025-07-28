"""
Latent Geometry Analysis - Part 1
Market Manifold Geometric Structure Analysis

This script analyzes the geometric properties of the latent space learned by the VAE:
- Computes decoder Jacobians and Riemannian metric tensors
- Approximates geodesic distances vs Euclidean distances
- Performs clustering in both spaces
- Creates 3D interactive visualizations of the manifold structure
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, permutation_test
from datetime import datetime
import warnings
from model import SimpleVAE

# Try to import plotly for 3D interactive visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    print("⚠️ Plotly not found. 3D interactive visualizations will be skipped.")
    print("   Install with: pip install plotly")
    HAS_PLOTLY = False

warnings.filterwarnings("ignore")


class LatentGeometryAnalyzer:
    """Analyze the geometric structure of the learned latent manifold."""

    def __init__(
        self, results_path: str, output_dir: str = None, n_samples: int = 1000
    ):
        """
        Initialize the geometry analyzer.

        Args:
            results_path: Path to VAE training results
            output_dir: Output directory for geometry analysis
            n_samples: Number of samples to use for analysis (to avoid O(n²) explosion)
        """
        self.results_path = Path(results_path)
        self.output_dir = (
            Path(output_dir) if output_dir else self.results_path / "geometry_analysis"
        )
        self.output_dir.mkdir(exist_ok=True)
        self.n_samples = n_samples

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🧠 Latent Geometry Analyzer initialized")
        print(f"📁 Results path: {self.results_path}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🔢 Using {n_samples} samples for analysis")
        print(f"🖥️  Device: {self.device}")

        # Load model and data
        self._load_model_and_data()

    def _load_model_and_data(self):
        """Load the trained model and latent representations."""
        print("\n📦 Loading model and latent data...")

        try:
            # Load config first
            with open(self.results_path / "training_config.json", "r") as f:
                self.config = json.load(f)

            self.latent_dim = self.config["model_config"]["latent_dim"]
            self.input_dim = self.config["model_config"]["input_dim"]
            hidden_dim = self.config["model_config"]["hidden_dim"]
            dropout_rate = self.config["model_config"]["dropout_rate"]

            # Reconstruct the model from config
            self.model = SimpleVAE(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                latent_dim=self.latent_dim,
                dropout_rate=dropout_rate,
            ).to(self.device)

            # Load the trained weights
            state_dict = torch.load(
                self.results_path / "model_weights.pth", map_location=self.device
            )
            self.model.load_state_dict(state_dict)
            self.model.eval()

            # Load latent vectors
            train_latent = np.load(self.results_path / "train_latent.npy")
            val_latent = np.load(self.results_path / "val_latent.npy")
            test_latent = np.load(self.results_path / "test_latent.npy")

            # Combine all latent vectors
            all_latent = np.vstack([train_latent, val_latent, test_latent])

            # Load date information
            try:
                with open(self.results_path / "date_info.json", "r") as f:
                    date_info = json.load(f)
                all_dates = pd.to_datetime(
                    date_info["train_dates"]
                    + date_info["val_dates"]
                    + date_info["test_dates"]
                )
                self.has_dates = True
            except FileNotFoundError:
                print(
                    "   ⚠️  Warning: date_info.json not found. Dates will not be included."
                )
                all_dates = None
                self.has_dates = False

            # Subsample if too large
            if len(all_latent) > self.n_samples:
                # Sort indices to maintain temporal order for plots
                indices = np.sort(
                    np.random.choice(len(all_latent), self.n_samples, replace=False)
                )
                self.z = all_latent[indices]
                self.sample_indices = indices
                if self.has_dates:
                    self.dates = all_dates[indices]
                print(f"🔢 Subsampled to {self.n_samples} points")
            else:
                self.z = all_latent
                self.sample_indices = np.arange(len(all_latent))
                if self.has_dates:
                    self.dates = all_dates
                print(f"🔢 Using all {len(all_latent)} points")

            # Convert to tensor
            self.z_tensor = torch.from_numpy(self.z).float().to(self.device)

            # Create time indices (for temporal analysis)
            self.time_indices = np.arange(len(self.z))

            print(f"✅ Loaded model and data")
            print(f"🧠 Model: {self.model.__class__.__name__}")
            print(f"🧠 Latent shape: {self.z.shape}")
            print(f"📊 Latent dim: {self.latent_dim}, Input dim: {self.input_dim}")

        except Exception as e:
            print(f"❌ Error loading model/data: {e}")
            raise

    def compute_jacobians(self):
        """Compute decoder Jacobians for each latent point."""
        print("\n🔄 Computing decoder Jacobians...")

        n_samples = len(self.z_tensor)
        jacobians = np.zeros((n_samples, self.input_dim, self.latent_dim))

        self.model.eval()

        for i, z_point in enumerate(self.z_tensor):
            if i % 100 == 0:
                print(f"   Processing sample {i+1}/{n_samples}")

            # Enable gradients for this point
            z_point = z_point.clone().requires_grad_(True)

            # Forward pass through decoder only
            with torch.enable_grad():
                # Get decoder output
                x_recon = self.model.decoder(z_point.unsqueeze(0))
                x_recon = x_recon.squeeze()

                # Compute Jacobian: ∂decoder(z)/∂z
                jacobian = torch.zeros(self.input_dim, self.latent_dim)

                for j in range(self.input_dim):
                    grad_outputs = torch.zeros_like(x_recon)
                    grad_outputs[j] = 1.0

                    grads = torch.autograd.grad(
                        outputs=x_recon,
                        inputs=z_point,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        create_graph=False,
                    )[0]

                    jacobian[j] = grads

                jacobians[i] = jacobian.detach().cpu().numpy()

        self.jacobians = jacobians
        print(f"✅ Computed Jacobians: shape {self.jacobians.shape}")

        # Save jacobians
        np.save(self.output_dir / "jacobians.npy", self.jacobians)

        return self.jacobians

    def compute_riemannian_metrics(self):
        """Compute Riemannian metric tensors G(z) = J(z)^T * J(z)."""
        print("\n📐 Computing Riemannian metric tensors...")

        if not hasattr(self, "jacobians"):
            self.compute_jacobians()

        n_samples = len(self.jacobians)
        self.metrics = np.zeros((n_samples, self.latent_dim, self.latent_dim))

        for i in range(n_samples):
            J = self.jacobians[i]  # [input_dim, latent_dim]
            G = J.T @ J  # [latent_dim, latent_dim]
            self.metrics[i] = G

        print(f"✅ Computed Riemannian metrics: shape {self.metrics.shape}")

        # Compute curvature measures
        self.curvatures = np.array([np.trace(G) for G in self.metrics])
        self.jacobian_norms = np.array(
            [np.linalg.norm(J, "fro") for J in self.jacobians]
        )

        # Save metrics
        np.save(self.output_dir / "riemannian_metrics.npy", self.metrics)
        np.save(self.output_dir / "curvatures.npy", self.curvatures)
        np.save(self.output_dir / "jacobian_norms.npy", self.jacobian_norms)

        return self.metrics

    def compute_geodesic_distances(self):
        """Approximate geodesic distances using Riemannian metrics."""
        print("\n🌐 Computing geodesic distances...")

        if not hasattr(self, "metrics"):
            self.compute_riemannian_metrics()

        n_samples = len(self.z)
        self.geodesic_distances = np.zeros((n_samples, n_samples))

        print("   Computing pairwise geodesic distances...")
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"   Processing row {i+1}/{n_samples}")

            for j in range(i, n_samples):
                if i == j:
                    self.geodesic_distances[i, j] = 0.0
                else:
                    # Difference vector
                    dz = self.z[i] - self.z[j]

                    # Use metric at point i for approximation
                    G_i = self.metrics[i]

                    # Geodesic distance approximation: sqrt((z_i - z_j)^T * G(z_i) * (z_i - z_j))
                    geodesic_dist = np.sqrt(dz.T @ G_i @ dz)

                    self.geodesic_distances[i, j] = geodesic_dist
                    self.geodesic_distances[j, i] = geodesic_dist

        # Also compute Euclidean distances for comparison
        self.euclidean_distances = cdist(self.z, self.z)

        print(f"✅ Computed geodesic and Euclidean distance matrices")

        # Save distances
        np.save(self.output_dir / "geodesic_distances.npy", self.geodesic_distances)
        np.save(self.output_dir / "euclidean_distances.npy", self.euclidean_distances)

        return self.geodesic_distances, self.euclidean_distances

    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering in both Euclidean and geodesic spaces."""
        print(f"\n🎯 Performing clustering with k={n_clusters}...")

        if not hasattr(self, "geodesic_distances"):
            self.compute_geodesic_distances()

        # Euclidean clustering (direct on latent space)
        kmeans_euclidean = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels_euclidean = kmeans_euclidean.fit_predict(self.z)

        # Geodesic clustering (use MDS to embed geodesic distances, then cluster)
        print("   Embedding geodesic distances with MDS...")
        mds = MDS(
            n_components=self.latent_dim, dissimilarity="precomputed", random_state=42
        )
        z_geodesic = mds.fit_transform(self.geodesic_distances)

        kmeans_geodesic = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels_geodesic = kmeans_geodesic.fit_predict(z_geodesic)

        print(f"✅ Clustering complete")
        print(f"   Euclidean clusters: {len(np.unique(self.labels_euclidean))}")
        print(f"   Geodesic clusters: {len(np.unique(self.labels_geodesic))}")

        # Save clustering results
        np.save(self.output_dir / "euclidean_clusters.npy", self.labels_euclidean)
        np.save(self.output_dir / "geodesic_clusters.npy", self.labels_geodesic)

        # Compare clustering quality
        self.compare_clustering_quality(z_geodesic)

        return self.labels_euclidean, self.labels_geodesic

    def compare_clustering_quality(self, z_geodesic):
        """Compare clustering quality between Euclidean and geodesic approaches."""
        print("\n📊 Comparing clustering quality...")

        # Calculate clustering quality metrics for both approaches with safety checks
        try:
            euclidean_metrics = {
                "silhouette": silhouette_score(self.z, self.labels_euclidean),
                "calinski_harabasz": calinski_harabasz_score(
                    self.z, self.labels_euclidean
                ),
                "davies_bouldin": davies_bouldin_score(self.z, self.labels_euclidean),
            }
        except Exception as e:
            print(f"   ⚠️  Warning: Error calculating Euclidean metrics: {e}")
            euclidean_metrics = {
                "silhouette": 0.0,
                "calinski_harabasz": 0.0,
                "davies_bouldin": 1.0,
            }

        try:
            geodesic_metrics = {
                "silhouette": silhouette_score(z_geodesic, self.labels_geodesic),
                "calinski_harabasz": calinski_harabasz_score(
                    z_geodesic, self.labels_geodesic
                ),
                "davies_bouldin": davies_bouldin_score(
                    z_geodesic, self.labels_geodesic
                ),
            }
        except Exception as e:
            print(f"   ⚠️  Warning: Error calculating geodesic metrics: {e}")
            geodesic_metrics = {
                "silhouette": 0.0,
                "calinski_harabasz": 0.0,
                "davies_bouldin": 1.0,
            }

        # Calculate cluster agreement
        cluster_agreement = adjusted_rand_score(
            self.labels_euclidean, self.labels_geodesic
        )

        # Perform bootstrap test for silhouette score difference
        def silhouette_diff(labels_euclidean, labels_geodesic, z_euclidean, z_geodesic):
            """Calculate difference in silhouette scores."""
            sil_euclidean = silhouette_score(z_euclidean, labels_euclidean)
            sil_geodesic = silhouette_score(z_geodesic, labels_geodesic)
            return sil_geodesic - sil_euclidean

        # Bootstrap test for statistical significance
        n_bootstrap = 1000
        bootstrap_diffs = []

        for _ in range(n_bootstrap):
            # Resample indices
            indices = np.random.choice(len(self.z), len(self.z), replace=True)

            # Calculate silhouette difference for this bootstrap sample
            z_boot = self.z[indices]
            z_geo_boot = z_geodesic[indices]
            labels_euc_boot = self.labels_euclidean[indices]
            labels_geo_boot = self.labels_geodesic[indices]

            try:
                diff = silhouette_diff(
                    labels_euc_boot, labels_geo_boot, z_boot, z_geo_boot
                )
                bootstrap_diffs.append(diff)
            except:
                continue  # Skip if bootstrap sample has issues

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate p-value (two-tailed test) with safety checks
        observed_diff = euclidean_metrics["silhouette"] - geodesic_metrics["silhouette"]

        if len(bootstrap_diffs) == 0:
            print("   ⚠️  Warning: Bootstrap test failed, using p-value = 1.0")
            p_value = 1.0
        else:
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
            print(f"   📊 Bootstrap samples used: {len(bootstrap_diffs)}/{n_bootstrap}")

        # Determine which clustering is better
        euclidean_better = 0
        geodesic_better = 0

        # Silhouette: higher is better
        if euclidean_metrics["silhouette"] > geodesic_metrics["silhouette"]:
            euclidean_better += 1
        else:
            geodesic_better += 1

        # Calinski-Harabasz: higher is better
        if (
            euclidean_metrics["calinski_harabasz"]
            > geodesic_metrics["calinski_harabasz"]
        ):
            euclidean_better += 1
        else:
            geodesic_better += 1

        # Davies-Bouldin: lower is better
        if euclidean_metrics["davies_bouldin"] < geodesic_metrics["davies_bouldin"]:
            euclidean_better += 1
        else:
            geodesic_better += 1

        # Store clustering comparison results (convert numpy types to native Python types)
        self.clustering_comparison = {
            "euclidean_metrics": {k: float(v) for k, v in euclidean_metrics.items()},
            "geodesic_metrics": {k: float(v) for k, v in geodesic_metrics.items()},
            "cluster_agreement": float(cluster_agreement),
            "statistical_test": {
                "observed_silhouette_difference": float(observed_diff),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            },
            "summary": {
                "euclidean_wins": int(euclidean_better),
                "geodesic_wins": int(geodesic_better),
                "better_approach": (
                    "Euclidean" if euclidean_better > geodesic_better else "Geodesic"
                ),
            },
        }

        # Save clustering comparison
        with open(self.output_dir / "clustering_comparison.json", "w") as f:
            json.dump(self.clustering_comparison, f, indent=2)

        # Create comparison visualization
        self.plot_clustering_comparison()

        # Print results with more detailed debugging
        print(f"✅ Clustering quality comparison complete")
        print(f"   📊 Euclidean Metrics:")
        for metric, value in euclidean_metrics.items():
            print(f"       {metric}: {value:.4f}")
        print(f"   📊 Geodesic Metrics:")
        for metric, value in geodesic_metrics.items():
            print(f"       {metric}: {value:.4f}")
        print(f"   📊 Cluster Agreement (ARI): {cluster_agreement:.4f}")
        print(
            f"   📊 Statistical significance (p={p_value:.4f}): {'Yes' if p_value < 0.05 else 'No'}"
        )
        print(
            f"   🏆 Better approach: {self.clustering_comparison['summary']['better_approach']}"
        )

        return self.clustering_comparison

    def plot_clustering_comparison(self):
        """Create visualization comparing clustering quality metrics."""
        print("   📊 Creating clustering comparison plot...")

        metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
        euclidean_values = [
            self.clustering_comparison["euclidean_metrics"][m] for m in metrics
        ]
        geodesic_values = [
            self.clustering_comparison["geodesic_metrics"][m] for m in metrics
        ]

        # Debug: Print raw values
        print(f"   🔍 Raw Euclidean values: {euclidean_values}")
        print(f"   🔍 Raw Geodesic values: {geodesic_values}")

        # Handle negative silhouette scores by shifting to positive range
        min_silhouette = min(euclidean_values[0], geodesic_values[0])
        if min_silhouette < 0:
            print(
                f"   ⚠️  Negative silhouette detected, shifting by {abs(min_silhouette)}"
            )
            euclidean_values[0] = euclidean_values[0] + abs(min_silhouette) + 0.1
            geodesic_values[0] = geodesic_values[0] + abs(min_silhouette) + 0.1

        # Normalize Davies-Bouldin (lower is better) for comparison
        # We'll invert it so higher bars are better for all metrics
        euclidean_values[2] = 1 / euclidean_values[2] if euclidean_values[2] > 0 else 0
        geodesic_values[2] = 1 / geodesic_values[2] if geodesic_values[2] > 0 else 0

        # Debug: Print processed values
        print(f"   🔍 Processed Euclidean values: {euclidean_values}")
        print(f"   🔍 Processed Geodesic values: {geodesic_values}")

        # Create separate subplots for each metric to handle scaling issues
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Silhouette Score (can be negative)
        sil_euc = self.clustering_comparison["euclidean_metrics"]["silhouette"]
        sil_geo = self.clustering_comparison["geodesic_metrics"]["silhouette"]

        axes[0, 0].bar(
            ["Euclidean", "Geodesic"],
            [sil_euc, sil_geo],
            color=["steelblue", "orange"],
            alpha=0.8,
        )
        axes[0, 0].set_title("Silhouette Score")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)

        # Add values as text
        for i, (name, val) in enumerate(
            [("Euclidean", sil_euc), ("Geodesic", sil_geo)]
        ):
            axes[0, 0].text(
                i,
                val + 0.01 if val >= 0 else val - 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontweight="bold",
            )

        # 2. Calinski-Harabasz Index (higher is better)
        ch_euc = self.clustering_comparison["euclidean_metrics"]["calinski_harabasz"]
        ch_geo = self.clustering_comparison["geodesic_metrics"]["calinski_harabasz"]

        axes[0, 1].bar(
            ["Euclidean", "Geodesic"],
            [ch_euc, ch_geo],
            color=["steelblue", "orange"],
            alpha=0.8,
        )
        axes[0, 1].set_title("Calinski-Harabasz Index")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].grid(True, alpha=0.3)

        # Add values as text
        for i, (name, val) in enumerate([("Euclidean", ch_euc), ("Geodesic", ch_geo)]):
            axes[0, 1].text(
                i,
                val + max(ch_euc, ch_geo) * 0.02,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. Davies-Bouldin Index (lower is better)
        db_euc = self.clustering_comparison["euclidean_metrics"]["davies_bouldin"]
        db_geo = self.clustering_comparison["geodesic_metrics"]["davies_bouldin"]

        axes[1, 0].bar(
            ["Euclidean", "Geodesic"],
            [db_euc, db_geo],
            color=["steelblue", "orange"],
            alpha=0.8,
        )
        axes[1, 0].set_title("Davies-Bouldin Index (lower = better)")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].grid(True, alpha=0.3)

        # Add values as text
        for i, (name, val) in enumerate([("Euclidean", db_euc), ("Geodesic", db_geo)]):
            axes[1, 0].text(
                i,
                val + max(db_euc, db_geo) * 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Summary and statistical test
        better_approach = self.clustering_comparison["summary"]["better_approach"]
        p_value = self.clustering_comparison["statistical_test"]["p_value"]
        significance = "significant" if p_value < 0.05 else "not significant"

        # Create summary text
        axes[1, 1].axis("off")
        summary_text = f"""Clustering Quality Comparison

Winner: {better_approach}
p-value: {p_value:.4f} ({significance})

Metric Scores:
                    Euclidean   Geodesic
Silhouette:          {sil_euc:.3f}      {sil_geo:.3f}
Calinski-Harabasz:   {ch_euc:.1f}     {ch_geo:.1f}
Davies-Bouldin:      {db_euc:.3f}      {db_geo:.3f}

Cluster Agreement (ARI): {self.clustering_comparison['cluster_agreement']:.3f}
"""

        axes[1, 1].text(
            0.1,
            0.9,
            summary_text,
            transform=axes[1, 1].transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "clustering_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("   ✅ Saved clustering_comparison.png")

    def create_3d_visualizations(self):
        """Create 3D interactive visualizations of the latent manifold."""
        print("\n🎨 Creating 3D interactive visualizations...")

        if not HAS_PLOTLY:
            print("⚠️ Skipping 3D interactive visualizations (plotly not available)")
            print("   Creating alternative matplotlib 3D plots...")
            self.create_matplotlib_3d_plots()
            return

        if not hasattr(self, "labels_euclidean"):
            self.perform_clustering()

        # Use first 3 latent dimensions for 3D plotting
        z_3d = self.z[:, :3]

        # 1. 3D Latent Flow (colored by time)
        fig_flow = go.Figure(
            data=go.Scatter3d(
                x=z_3d[:, 0],
                y=z_3d[:, 1],
                z=z_3d[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=self.time_indices,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Time Index"),
                    opacity=0.8,
                ),
                text=[
                    f"Time: {t}<br>Curvature: {c:.3f}"
                    for t, c in zip(self.time_indices, self.curvatures)
                ],
                hovertemplate="<b>Time:</b> %{text}<br>"
                + "<b>Z1:</b> %{x:.3f}<br>"
                + "<b>Z2:</b> %{y:.3f}<br>"
                + "<b>Z3:</b> %{z:.3f}<extra></extra>",
            )
        )

        fig_flow.update_layout(
            title="3D Latent Flow (Colored by Time)",
            scene=dict(
                xaxis_title="Latent Dimension 1",
                yaxis_title="Latent Dimension 2",
                zaxis_title="Latent Dimension 3",
            ),
            width=800,
            height=600,
        )

        fig_flow.write_html(self.output_dir / "3d_latent_flow.html")
        print("   ✅ Saved 3d_latent_flow.html")

        # 2. 3D Curvature Map
        fig_curvature = go.Figure(
            data=go.Scatter3d(
                x=z_3d[:, 0],
                y=z_3d[:, 1],
                z=z_3d[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=self.curvatures,
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(title="Trace(G(z))"),
                    opacity=0.8,
                ),
                text=[
                    f"Curvature: {c:.3f}<br>JacNorm: {jn:.3f}"
                    for c, jn in zip(self.curvatures, self.jacobian_norms)
                ],
                hovertemplate="<b>Curvature:</b> %{text}<br>"
                + "<b>Z1:</b> %{x:.3f}<br>"
                + "<b>Z2:</b> %{y:.3f}<br>"
                + "<b>Z3:</b> %{z:.3f}<extra></extra>",
            )
        )

        fig_curvature.update_layout(
            title="3D Curvature Map (Colored by Trace(G(z)))",
            scene=dict(
                xaxis_title="Latent Dimension 1",
                yaxis_title="Latent Dimension 2",
                zaxis_title="Latent Dimension 3",
            ),
            width=800,
            height=600,
        )

        fig_curvature.write_html(self.output_dir / "3d_curvature_map.html")
        print("   ✅ Saved 3d_curvature_map.html")

        # 3. 3D Clustering Comparison
        fig_clusters = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Euclidean Clustering", "Geodesic Clustering"],
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        )

        # Euclidean clusters
        colors_euclidean = px.colors.qualitative.Set1[
            : len(np.unique(self.labels_euclidean))
        ]
        for i, cluster in enumerate(np.unique(self.labels_euclidean)):
            mask = self.labels_euclidean == cluster
            fig_clusters.add_trace(
                go.Scatter3d(
                    x=z_3d[mask, 0],
                    y=z_3d[mask, 1],
                    z=z_3d[mask, 2],
                    mode="markers",
                    marker=dict(size=3, color=colors_euclidean[i], opacity=0.7),
                    name=f"E-Cluster {cluster}",
                    legendgroup="euclidean",
                ),
                row=1,
                col=1,
            )

        # Geodesic clusters
        colors_geodesic = px.colors.qualitative.Set2[
            : len(np.unique(self.labels_geodesic))
        ]
        for i, cluster in enumerate(np.unique(self.labels_geodesic)):
            mask = self.labels_geodesic == cluster
            fig_clusters.add_trace(
                go.Scatter3d(
                    x=z_3d[mask, 0],
                    y=z_3d[mask, 1],
                    z=z_3d[mask, 2],
                    mode="markers",
                    marker=dict(size=3, color=colors_geodesic[i], opacity=0.7),
                    name=f"G-Cluster {cluster}",
                    legendgroup="geodesic",
                ),
                row=1,
                col=2,
            )

        fig_clusters.update_layout(
            title="3D Clustering Comparison: Euclidean vs Geodesic",
            width=1400,
            height=600,
        )

        fig_clusters.write_html(self.output_dir / "3d_clustering_comparison.html")
        print("   ✅ Saved 3d_clustering_comparison.html")

        # 4. Jacobian Norm Map
        fig_jacobian = go.Figure(
            data=go.Scatter3d(
                x=z_3d[:, 0],
                y=z_3d[:, 1],
                z=z_3d[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=self.jacobian_norms,
                    colorscale="Inferno",
                    showscale=True,
                    colorbar=dict(title="||J(z)||_F"),
                    opacity=0.8,
                ),
                text=[
                    f"JacNorm: {jn:.3f}<br>Curvature: {c:.3f}"
                    for jn, c in zip(self.jacobian_norms, self.curvatures)
                ],
                hovertemplate="<b>Jacobian Norm:</b> %{text}<br>"
                + "<b>Z1:</b> %{x:.3f}<br>"
                + "<b>Z2:</b> %{y:.3f}<br>"
                + "<b>Z3:</b> %{z:.3f}<extra></extra>",
            )
        )

        fig_jacobian.update_layout(
            title="3D Jacobian Norm Map (Colored by ||J(z)||)",
            scene=dict(
                xaxis_title="Latent Dimension 1",
                yaxis_title="Latent Dimension 2",
                zaxis_title="Latent Dimension 3",
            ),
            width=800,
            height=600,
        )

        fig_jacobian.write_html(self.output_dir / "3d_jacobian_norm_map.html")
        print("   ✅ Saved 3d_jacobian_norm_map.html")

    def create_matplotlib_3d_plots(self):
        """Create 3D plots using matplotlib as fallback when plotly is not available."""
        print("\n🎨 Creating matplotlib 3D plots...")

        if not hasattr(self, "labels_euclidean"):
            self.perform_clustering()

        # Use first 3 latent dimensions for 3D plotting
        z_3d = self.z[:, :3]

        # Create 3D subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. 3D Latent Flow (colored by time)
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        scatter1 = ax1.scatter(
            z_3d[:, 0],
            z_3d[:, 1],
            z_3d[:, 2],
            c=self.time_indices,
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        ax1.set_title("3D Latent Flow (Colored by Time)")
        ax1.set_xlabel("Latent Dimension 1")
        ax1.set_ylabel("Latent Dimension 2")
        ax1.set_zlabel("Latent Dimension 3")
        plt.colorbar(scatter1, ax=ax1, label="Time Index", shrink=0.8)

        # 2. 3D Curvature Map
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        scatter2 = ax2.scatter(
            z_3d[:, 0],
            z_3d[:, 1],
            z_3d[:, 2],
            c=self.curvatures,
            cmap="plasma",
            s=10,
            alpha=0.7,
        )
        ax2.set_title("3D Curvature Map (Colored by Trace(G(z)))")
        ax2.set_xlabel("Latent Dimension 1")
        ax2.set_ylabel("Latent Dimension 2")
        ax2.set_zlabel("Latent Dimension 3")
        plt.colorbar(scatter2, ax=ax2, label="Trace(G(z))", shrink=0.8)

        # 3. 3D Euclidean Clustering
        ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        scatter3 = ax3.scatter(
            z_3d[:, 0],
            z_3d[:, 1],
            z_3d[:, 2],
            c=self.labels_euclidean,
            cmap="Set1",
            s=10,
            alpha=0.7,
        )
        ax3.set_title("3D Euclidean Clustering")
        ax3.set_xlabel("Latent Dimension 1")
        ax3.set_ylabel("Latent Dimension 2")
        ax3.set_zlabel("Latent Dimension 3")

        # 4. 3D Jacobian Norm Map
        ax4 = fig.add_subplot(2, 2, 4, projection="3d")
        scatter4 = ax4.scatter(
            z_3d[:, 0],
            z_3d[:, 1],
            z_3d[:, 2],
            c=self.jacobian_norms,
            cmap="inferno",
            s=10,
            alpha=0.7,
        )
        ax4.set_title("3D Jacobian Norm Map")
        ax4.set_xlabel("Latent Dimension 1")
        ax4.set_ylabel("Latent Dimension 2")
        ax4.set_zlabel("Latent Dimension 3")
        plt.colorbar(scatter4, ax=ax4, label="||J(z)||_F", shrink=0.8)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "3d_matplotlib_plots.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("   ✅ Saved 3d_matplotlib_plots.png")

    def create_2d_analysis_plots(self):
        """Create 2D analysis plots for comparison."""
        print("\n📊 Creating 2D analysis plots...")

        # 1. Geodesic vs Euclidean distance comparison
        plt.figure(figsize=(10, 8))

        # Sample points for scatter plot (avoid too many points)
        n_sample = min(1000, len(self.geodesic_distances))
        idx = np.random.choice(len(self.geodesic_distances), n_sample, replace=False)

        geo_flat = self.geodesic_distances[np.ix_(idx, idx)].flatten()
        euc_flat = self.euclidean_distances[np.ix_(idx, idx)].flatten()

        plt.scatter(euc_flat, geo_flat, alpha=0.5, s=1)
        plt.plot(
            [0, max(euc_flat)], [0, max(geo_flat)], "r--", alpha=0.7, label="y = x"
        )
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Geodesic Distance")
        plt.title("Geodesic vs Euclidean Distances")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "geodesic_vs_euclidean.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Riemannian trace time series
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_indices, self.curvatures, alpha=0.7, linewidth=1)
        plt.xlabel("Time Index")
        plt.ylabel("Trace(G(z))")
        plt.title("Riemannian Metric Trace Over Time")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "riemannian_trace_timeseries.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("   ✅ Saved 2D analysis plots")

    def save_summary_data(self):
        """Save summary data and create comprehensive report."""
        print("\n💾 Saving summary data...")

        # Create summary dataframe
        summary_data = {
            "Time_Index": self.time_indices,
            "Original_Index": self.sample_indices,
            "Latent_1": self.z[:, 0],
            "Latent_2": self.z[:, 1],
            "Latent_3": self.z[:, 2],
            "Curvature": self.curvatures,
            "Jacobian_Norm": self.jacobian_norms,
            "Euclidean_Cluster": self.labels_euclidean,
            "Geodesic_Cluster": self.labels_geodesic,
        }
        if self.has_dates:
            summary_data["Date"] = self.dates.strftime("%Y-%m-%d")

        summary_df = pd.DataFrame(summary_data)

        # Add all latent dimensions
        for i in range(self.latent_dim):
            if f"Latent_{i+1}" not in summary_df.columns:
                summary_df[f"Latent_{i+1}"] = self.z[:, i]

        summary_df.to_csv(self.output_dir / "latent_geometry_summary.csv", index=False)

        # Compute summary statistics (ensure all values are JSON serializable)
        stats = {
            "analysis_timestamp": datetime.now().isoformat(),
            "n_samples_analyzed": int(len(self.z)),
            "latent_dimension": int(self.latent_dim),
            "input_dimension": int(self.input_dim),
            "curvature_stats": {
                "mean": float(np.mean(self.curvatures)),
                "std": float(np.std(self.curvatures)),
                "min": float(np.min(self.curvatures)),
                "max": float(np.max(self.curvatures)),
            },
            "jacobian_norm_stats": {
                "mean": float(np.mean(self.jacobian_norms)),
                "std": float(np.std(self.jacobian_norms)),
                "min": float(np.min(self.jacobian_norms)),
                "max": float(np.max(self.jacobian_norms)),
            },
            "distance_correlation": float(
                np.corrcoef(
                    self.geodesic_distances.flatten(),
                    self.euclidean_distances.flatten(),
                )[0, 1]
            ),
            "clustering_comparison": (
                self.clustering_comparison
                if hasattr(self, "clustering_comparison")
                else None
            ),
        }

        with open(self.output_dir / "geometry_analysis_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✅ Saved summary data")
        print(
            f"   📊 Curvature range: [{stats['curvature_stats']['min']:.3f}, {stats['curvature_stats']['max']:.3f}]"
        )
        print(
            f"   📊 Geodesic-Euclidean correlation: {stats['distance_correlation']:.3f}"
        )

        if hasattr(self, "clustering_comparison"):
            better_approach = self.clustering_comparison["summary"]["better_approach"]
            p_value = self.clustering_comparison["statistical_test"]["p_value"]
            print(f"   🏆 Better clustering: {better_approach} (p={p_value:.4f})")

    def run_full_analysis(self, n_clusters=5):
        """Run the complete latent geometry analysis."""
        print("\n🚀 Running full latent geometry analysis...")

        # Step 1: Compute geometric properties
        self.compute_jacobians()
        self.compute_riemannian_metrics()
        self.compute_geodesic_distances()

        # Step 2: Perform clustering
        self.perform_clustering(n_clusters=n_clusters)

        # Step 3: Create visualizations
        self.create_3d_visualizations()
        self.create_2d_analysis_plots()

        # Step 4: Save summary
        self.save_summary_data()

        print(f"\n🎉 Latent geometry analysis complete!")
        print(f"📁 All results saved to: {self.output_dir}")

        if HAS_PLOTLY:
            print(f"🌐 Interactive 3D plots:")
            print(f"   - {self.output_dir}/3d_latent_flow.html")
            print(f"   - {self.output_dir}/3d_curvature_map.html")
            print(f"   - {self.output_dir}/3d_clustering_comparison.html")
            print(f"   - {self.output_dir}/3d_jacobian_norm_map.html")
        else:
            print(f"📊 Matplotlib 3D plots:")
            print(f"   - {self.output_dir}/3d_matplotlib_plots.png")
            print(f"💡 For interactive 3D plots, install plotly: pip install plotly")

        print(f"📊 Additional analyses:")
        print(f"   - {self.output_dir}/clustering_comparison.png")
        print(f"   - {self.output_dir}/clustering_comparison.json")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Latent Geometry Analysis")
    parser.add_argument("results_path", help="Path to VAE training results folder")
    parser.add_argument(
        "--output_dir", help="Output directory for geometry analysis", default=None
    )
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples to analyze"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=5, help="Number of clusters for K-means"
    )

    args = parser.parse_args()

    # Create analyzer and run full analysis
    analyzer = LatentGeometryAnalyzer(
        results_path=args.results_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
    )

    analyzer.run_full_analysis(n_clusters=args.n_clusters)


if __name__ == "__main__":
    main()
