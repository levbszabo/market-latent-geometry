"""
Latent Geometry Analysis - Part 1
Market Manifold Geometric Structure Analysis

This script analyzes the geometric properties of the latent space learned by the VAE:
- Computes decoder Jacobians and Riemannian metric tensors
- Approximates geodesic distances vs Euclidean distances
- Performs clustering in three spaces: Euclidean, Geodesic (MDS), and PCA-reduced
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
from sklearn.decomposition import PCA
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

# Additional imports for clustering visualization
from sklearn.neighbors import KNeighborsClassifier

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
        self.pca_dims = 5  # Number of dimensions for PCA clustering

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🧠 Latent Geometry Analyzer initialized")
        print(f"📁 Results path: {self.results_path}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🔢 Using {n_samples} samples for analysis")
        print(f"📐 PCA dimensions: {self.pca_dims}")
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
        """Perform K-means clustering in three spaces: Euclidean, Geodesic (MDS), and PCA-reduced."""
        print(f"\n🎯 Performing clustering with k={n_clusters} in three spaces...")

        if not hasattr(self, "geodesic_distances"):
            self.compute_geodesic_distances()

        # 1. Euclidean clustering (direct on latent space)
        print("   🔹 Euclidean clustering (direct on latent space)...")
        kmeans_euclidean = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels_euclidean = kmeans_euclidean.fit_predict(self.z)

        # Create MDS embedding of Euclidean distances for visualization
        print("      Computing MDS embedding of Euclidean distances...")
        mds_euclidean = MDS(
            n_components=2, dissimilarity="precomputed", random_state=42
        )
        self.z_euclidean_mds = mds_euclidean.fit_transform(self.euclidean_distances)

        # 2. Geodesic clustering (use MDS to embed geodesic distances, then cluster)
        print("   🔹 Geodesic clustering (MDS embedding)...")
        print("      Embedding geodesic distances with MDS...")
        mds = MDS(
            n_components=self.latent_dim, dissimilarity="precomputed", random_state=42
        )
        self.z_geodesic = mds.fit_transform(self.geodesic_distances)

        # Also create 2D and 3D versions for visualization
        mds_2d = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        self.z_geodesic_2d = mds_2d.fit_transform(self.geodesic_distances)

        mds_3d = MDS(n_components=3, dissimilarity="precomputed", random_state=42)
        self.z_geodesic_3d = mds_3d.fit_transform(self.geodesic_distances)

        kmeans_geodesic = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels_geodesic = kmeans_geodesic.fit_predict(self.z_geodesic)

        # 3. PCA clustering (reduce to 5D, then cluster with Euclidean distance)
        print(f"   🔹 PCA clustering (reduce to {self.pca_dims}D)...")
        self.pca = PCA(n_components=self.pca_dims, random_state=42)
        self.z_pca = self.pca.fit_transform(self.z)

        print(
            f"      PCA explained variance ratio: {self.pca.explained_variance_ratio_}"
        )
        print(
            f"      Total variance explained: {self.pca.explained_variance_ratio_.sum():.3f}"
        )

        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels_pca = kmeans_pca.fit_predict(self.z_pca)

        print(f"✅ Clustering complete")
        print(f"   Euclidean clusters: {len(np.unique(self.labels_euclidean))}")
        print(f"   Geodesic clusters: {len(np.unique(self.labels_geodesic))}")
        print(f"   PCA clusters: {len(np.unique(self.labels_pca))}")

        # Save clustering results
        np.save(self.output_dir / "euclidean_clusters.npy", self.labels_euclidean)
        np.save(self.output_dir / "geodesic_clusters.npy", self.labels_geodesic)
        np.save(self.output_dir / "pca_clusters.npy", self.labels_pca)
        np.save(self.output_dir / "pca_components.npy", self.pca.components_)
        np.save(
            self.output_dir / "pca_explained_variance.npy",
            self.pca.explained_variance_ratio_,
        )
        np.save(self.output_dir / "z_pca.npy", self.z_pca)
        np.save(self.output_dir / "z_euclidean_mds.npy", self.z_euclidean_mds)
        np.save(self.output_dir / "z_geodesic_2d.npy", self.z_geodesic_2d)
        np.save(self.output_dir / "z_geodesic_3d.npy", self.z_geodesic_3d)

        # Compare clustering quality
        self.compare_clustering_quality()

        return self.labels_euclidean, self.labels_geodesic, self.labels_pca

    def compare_clustering_quality(self):
        """Compare clustering quality between Euclidean, Geodesic, and PCA approaches."""
        print("\n📊 Comparing clustering quality across three approaches...")

        # Calculate clustering quality metrics for all three approaches with safety checks
        def safe_calculate_metrics(z_data, labels, approach_name):
            """Safely calculate clustering metrics with error handling."""
            try:
                metrics = {
                    "silhouette": silhouette_score(z_data, labels),
                    "calinski_harabasz": calinski_harabasz_score(z_data, labels),
                    "davies_bouldin": davies_bouldin_score(z_data, labels),
                }
                return metrics
            except Exception as e:
                print(f"   ⚠️  Warning: Error calculating {approach_name} metrics: {e}")
                return {
                    "silhouette": 0.0,
                    "calinski_harabasz": 0.0,
                    "davies_bouldin": 1.0,
                }

        euclidean_metrics = safe_calculate_metrics(
            self.z, self.labels_euclidean, "Euclidean"
        )
        # For geodesic clustering, use precomputed geodesic distances for silhouette score
        # but MDS embedding for other metrics (which require coordinate vectors)
        try:
            geodesic_metrics = {
                "silhouette": silhouette_score(
                    self.geodesic_distances, self.labels_geodesic, metric="precomputed"
                ),
                "calinski_harabasz": calinski_harabasz_score(
                    self.z_geodesic, self.labels_geodesic
                ),
                "davies_bouldin": davies_bouldin_score(
                    self.z_geodesic, self.labels_geodesic
                ),
            }
        except Exception as e:
            print(f"   ⚠️  Warning: Error calculating Geodesic metrics: {e}")
            geodesic_metrics = {
                "silhouette": 0.0,
                "calinski_harabasz": 0.0,
                "davies_bouldin": 1.0,
            }
        pca_metrics = safe_calculate_metrics(self.z_pca, self.labels_pca, "PCA")

        # Calculate pairwise cluster agreements
        agreement_euc_geo = adjusted_rand_score(
            self.labels_euclidean, self.labels_geodesic
        )
        agreement_euc_pca = adjusted_rand_score(self.labels_euclidean, self.labels_pca)
        agreement_geo_pca = adjusted_rand_score(self.labels_geodesic, self.labels_pca)

        # Perform permutation test for silhouette score differences
        def permutation_test_clustering(n_permutations=1000):
            """
            Permutation test using all data points.
            Tests null hypothesis: clustering method has no effect on clustering quality.

            H0: Silhouette scores from different methods are from same distribution
            H1: At least one method produces significantly different clustering quality
            """
            print(
                f"   🔬 Performing permutation tests with all data ({n_permutations} permutations)..."
            )

            # Get baseline silhouette scores (observed test statistics)
            sil_euclidean = euclidean_metrics["silhouette"]
            sil_geodesic = geodesic_metrics["silhouette"]
            sil_pca = pca_metrics["silhouette"]

            # Observed differences (our test statistics)
            obs_diff_euc_geo = sil_geodesic - sil_euclidean
            obs_diff_euc_pca = sil_pca - sil_euclidean
            obs_diff_geo_pca = sil_pca - sil_geodesic

            print(f"      Observed differences:")
            print(f"        Geodesic - Euclidean: {obs_diff_euc_geo:.4f}")
            print(f"        PCA - Euclidean: {obs_diff_euc_pca:.4f}")
            print(f"        PCA - Geodesic: {obs_diff_geo_pca:.4f}")

            # Permutation test: randomly assign labels and recompute
            perm_diffs_euc_geo = []
            perm_diffs_euc_pca = []
            perm_diffs_geo_pca = []

            for i in range(n_permutations):
                if (i + 1) % 200 == 0:
                    print(f"        Permutation {i+1}/{n_permutations}")

                # Randomly permute cluster labels (this breaks the clustering-performance link)
                perm_labels_euc = np.random.permutation(self.labels_euclidean)
                perm_labels_geo = np.random.permutation(self.labels_geodesic)
                perm_labels_pca = np.random.permutation(self.labels_pca)

                try:
                    # Compute silhouette scores with permuted labels
                    # If H0 is true, these should be similar to original scores
                    perm_sil_euc = silhouette_score(self.z, perm_labels_euc)
                    perm_sil_geo = silhouette_score(
                        self.geodesic_distances, perm_labels_geo, metric="precomputed"
                    )
                    perm_sil_pca = silhouette_score(self.z_pca, perm_labels_pca)

                    # Store differences under null hypothesis
                    perm_diffs_euc_geo.append(perm_sil_geo - perm_sil_euc)
                    perm_diffs_euc_pca.append(perm_sil_pca - perm_sil_euc)
                    perm_diffs_geo_pca.append(perm_sil_pca - perm_sil_geo)

                except Exception as e:
                    # Skip permutations that fail (e.g., all points in one cluster)
                    continue

            # Calculate p-values: fraction of permuted differences >= observed
            # Two-tailed test: |permuted| >= |observed|
            p_euc_geo = np.mean(np.abs(perm_diffs_euc_geo) >= np.abs(obs_diff_euc_geo))
            p_euc_pca = np.mean(np.abs(perm_diffs_euc_pca) >= np.abs(obs_diff_euc_pca))
            p_geo_pca = np.mean(np.abs(perm_diffs_geo_pca) >= np.abs(obs_diff_geo_pca))

            print(
                f"      Valid permutations: {len(perm_diffs_euc_geo)}/{n_permutations}"
            )

            return {
                "p_euclidean_vs_geodesic": p_euc_geo,
                "p_euclidean_vs_pca": p_euc_pca,
                "p_geodesic_vs_pca": p_geo_pca,
                "observed_differences": {
                    "euclidean_vs_geodesic": obs_diff_euc_geo,
                    "euclidean_vs_pca": obs_diff_euc_pca,
                    "geodesic_vs_pca": obs_diff_geo_pca,
                },
                "permutation_distributions": {
                    "euclidean_vs_geodesic": perm_diffs_euc_geo,
                    "euclidean_vs_pca": perm_diffs_euc_pca,
                    "geodesic_vs_pca": perm_diffs_geo_pca,
                },
            }

        # Run permutation test
        perm_results = permutation_test_clustering(n_permutations=1000)

        # Extract results for compatibility with existing code
        obs_diff_euc_geo = perm_results["observed_differences"]["euclidean_vs_geodesic"]
        obs_diff_euc_pca = perm_results["observed_differences"]["euclidean_vs_pca"]
        obs_diff_geo_pca = perm_results["observed_differences"]["geodesic_vs_pca"]

        p_value_euc_geo = perm_results["p_euclidean_vs_geodesic"]
        p_value_euc_pca = perm_results["p_euclidean_vs_pca"]
        p_value_geo_pca = perm_results["p_geodesic_vs_pca"]

        # Determine ranking based on multiple metrics
        def rank_approaches():
            """Rank the three approaches based on clustering quality metrics."""
            approaches = {
                "Euclidean": euclidean_metrics,
                "Geodesic": geodesic_metrics,
                "PCA": pca_metrics,
            }

            scores = {}
            for name, metrics in approaches.items():
                score = 0
                # Silhouette: higher is better
                score += metrics["silhouette"]
                # Calinski-Harabasz: higher is better (normalize)
                ch_values = [m["calinski_harabasz"] for m in approaches.values()]
                if max(ch_values) > 0:
                    score += metrics["calinski_harabasz"] / max(ch_values)
                # Davies-Bouldin: lower is better (invert)
                db_values = [m["davies_bouldin"] for m in approaches.values()]
                if max(db_values) > 0:
                    score += (max(db_values) - metrics["davies_bouldin"]) / max(
                        db_values
                    )

                scores[name] = score

            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        ranking = rank_approaches()

        # Store comprehensive clustering comparison results
        self.clustering_comparison = {
            "euclidean_metrics": {k: float(v) for k, v in euclidean_metrics.items()},
            "geodesic_metrics": {k: float(v) for k, v in geodesic_metrics.items()},
            "pca_metrics": {k: float(v) for k, v in pca_metrics.items()},
            "pca_info": {
                "n_components": int(self.pca_dims),
                "explained_variance_ratio": [
                    float(x) for x in self.pca.explained_variance_ratio_
                ],
                "total_variance_explained": float(
                    self.pca.explained_variance_ratio_.sum()
                ),
            },
            "cluster_agreements": {
                "euclidean_vs_geodesic": float(agreement_euc_geo),
                "euclidean_vs_pca": float(agreement_euc_pca),
                "geodesic_vs_pca": float(agreement_geo_pca),
            },
            "statistical_tests": {
                "test_method": "permutation_test",
                "null_hypothesis": "Clustering method has no effect on clustering quality",
                "alternative_hypothesis": "At least one method produces significantly different clustering quality",
                "n_permutations": 1000,
                "test_statistic": "silhouette_score_difference",
                "euclidean_vs_geodesic": {
                    "observed_silhouette_difference": float(obs_diff_euc_geo),
                    "p_value": float(p_value_euc_geo),
                    "significant_at_0.05": bool(p_value_euc_geo < 0.05),
                    "significant_at_0.01": bool(p_value_euc_geo < 0.01),
                    "significant_bonferroni": bool(
                        p_value_euc_geo < 0.05 / 3
                    ),  # Multiple comparison correction
                    "interpretation": (
                        "Geodesic superior"
                        if obs_diff_euc_geo > 0
                        else "Euclidean superior"
                    ),
                },
                "euclidean_vs_pca": {
                    "observed_silhouette_difference": float(obs_diff_euc_pca),
                    "p_value": float(p_value_euc_pca),
                    "significant_at_0.05": bool(p_value_euc_pca < 0.05),
                    "significant_at_0.01": bool(p_value_euc_pca < 0.01),
                    "significant_bonferroni": bool(p_value_euc_pca < 0.05 / 3),
                    "interpretation": (
                        "PCA superior" if obs_diff_euc_pca > 0 else "Euclidean superior"
                    ),
                },
                "geodesic_vs_pca": {
                    "observed_silhouette_difference": float(obs_diff_geo_pca),
                    "p_value": float(p_value_geo_pca),
                    "significant_at_0.05": bool(p_value_geo_pca < 0.05),
                    "significant_at_0.01": bool(p_value_geo_pca < 0.01),
                    "significant_bonferroni": bool(p_value_geo_pca < 0.05 / 3),
                    "interpretation": (
                        "PCA superior" if obs_diff_geo_pca > 0 else "Geodesic superior"
                    ),
                },
                "bonferroni_alpha": 0.05 / 3,
                "permutation_results": perm_results,  # Store full permutation results
            },
            "ranking": {
                "first": ranking[0][0],
                "second": ranking[1][0],
                "third": ranking[2][0],
                "scores": {name: float(score) for name, score in ranking},
            },
        }

        # Save clustering comparison
        with open(self.output_dir / "clustering_comparison.json", "w") as f:
            json.dump(self.clustering_comparison, f, indent=2)

        # Create comparison visualization
        self.plot_clustering_comparison()

        # Print comprehensive results
        print(f"✅ Clustering quality comparison complete")
        print(f"   📊 Euclidean Metrics:")
        for metric, value in euclidean_metrics.items():
            print(f"       {metric}: {value:.4f}")
        print(f"   📊 Geodesic Metrics:")
        for metric, value in geodesic_metrics.items():
            print(f"       {metric}: {value:.4f}")
        print(f"   📊 PCA Metrics:")
        for metric, value in pca_metrics.items():
            print(f"       {metric}: {value:.4f}")
        print(
            f"   📊 PCA Variance Explained: {self.pca.explained_variance_ratio_.sum():.3f}"
        )
        print(f"   📊 Cluster Agreements (ARI):")
        print(f"       Euclidean vs Geodesic: {agreement_euc_geo:.4f}")
        print(f"       Euclidean vs PCA: {agreement_euc_pca:.4f}")
        print(f"       Geodesic vs PCA: {agreement_geo_pca:.4f}")
        print(f"   🧪 Permutation Test Results (n=1000):")
        print(f"       Null Hypothesis: Clustering method has no effect on quality")
        print(
            f"       Euclidean vs Geodesic: p={p_value_euc_geo:.4f} {'***' if p_value_euc_geo < 0.001 else '**' if p_value_euc_geo < 0.01 else '*' if p_value_euc_geo < 0.05 else 'ns'}"
        )
        print(
            f"       Euclidean vs PCA: p={p_value_euc_pca:.4f} {'***' if p_value_euc_pca < 0.001 else '**' if p_value_euc_pca < 0.01 else '*' if p_value_euc_pca < 0.05 else 'ns'}"
        )
        print(
            f"       Geodesic vs PCA: p={p_value_geo_pca:.4f} {'***' if p_value_geo_pca < 0.001 else '**' if p_value_geo_pca < 0.01 else '*' if p_value_geo_pca < 0.05 else 'ns'}"
        )
        bonferroni_alpha = 0.05 / 3
        print(f"       Bonferroni-corrected α = {bonferroni_alpha:.4f}")
        any_significant = any(
            [
                p_value_euc_geo < bonferroni_alpha,
                p_value_euc_pca < bonferroni_alpha,
                p_value_geo_pca < bonferroni_alpha,
            ]
        )
        print(
            f"       After multiple comparison correction: {'Significant differences detected' if any_significant else 'No significant differences'}"
        )
        print(
            f"   🏆 Ranking: 1st={ranking[0][0]}, 2nd={ranking[1][0]}, 3rd={ranking[2][0]}"
        )

        return self.clustering_comparison

    def plot_clustering_comparison(self):
        """Create visualization comparing clustering quality metrics across all three approaches."""
        print("   📊 Creating comprehensive clustering comparison plot...")

        metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
        euclidean_values = [
            self.clustering_comparison["euclidean_metrics"][m] for m in metrics
        ]
        geodesic_values = [
            self.clustering_comparison["geodesic_metrics"][m] for m in metrics
        ]
        pca_values = [self.clustering_comparison["pca_metrics"][m] for m in metrics]

        # Create comprehensive figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        approaches = ["Euclidean", "Geodesic", "PCA"]
        colors = ["steelblue", "orange", "green"]

        # 1. Silhouette Score
        sil_values = [
            self.clustering_comparison["euclidean_metrics"]["silhouette"],
            self.clustering_comparison["geodesic_metrics"]["silhouette"],
            self.clustering_comparison["pca_metrics"]["silhouette"],
        ]

        bars1 = axes[0, 0].bar(approaches, sil_values, color=colors, alpha=0.8)
        axes[0, 0].set_title("Silhouette Score")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)

        # Add values as text
        for i, (bar, val) in enumerate(zip(bars1, sil_values)):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01 if val >= 0 else val - 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontweight="bold",
            )

        # 2. Calinski-Harabasz Index
        ch_values = [
            self.clustering_comparison["euclidean_metrics"]["calinski_harabasz"],
            self.clustering_comparison["geodesic_metrics"]["calinski_harabasz"],
            self.clustering_comparison["pca_metrics"]["calinski_harabasz"],
        ]

        bars2 = axes[0, 1].bar(approaches, ch_values, color=colors, alpha=0.8)
        axes[0, 1].set_title("Calinski-Harabasz Index")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].grid(True, alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars2, ch_values)):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                val + max(ch_values) * 0.02,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. Davies-Bouldin Index
        db_values = [
            self.clustering_comparison["euclidean_metrics"]["davies_bouldin"],
            self.clustering_comparison["geodesic_metrics"]["davies_bouldin"],
            self.clustering_comparison["pca_metrics"]["davies_bouldin"],
        ]

        bars3 = axes[0, 2].bar(approaches, db_values, color=colors, alpha=0.8)
        axes[0, 2].set_title("Davies-Bouldin Index (lower = better)")
        axes[0, 2].set_ylabel("Score")
        axes[0, 2].grid(True, alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars3, db_values)):
            axes[0, 2].text(
                bar.get_x() + bar.get_width() / 2,
                val + max(db_values) * 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Cluster Agreement Matrix
        agreement_matrix = np.array(
            [
                [
                    1.0,
                    self.clustering_comparison["cluster_agreements"][
                        "euclidean_vs_geodesic"
                    ],
                    self.clustering_comparison["cluster_agreements"][
                        "euclidean_vs_pca"
                    ],
                ],
                [
                    self.clustering_comparison["cluster_agreements"][
                        "euclidean_vs_geodesic"
                    ],
                    1.0,
                    self.clustering_comparison["cluster_agreements"]["geodesic_vs_pca"],
                ],
                [
                    self.clustering_comparison["cluster_agreements"][
                        "euclidean_vs_pca"
                    ],
                    self.clustering_comparison["cluster_agreements"]["geodesic_vs_pca"],
                    1.0,
                ],
            ]
        )

        im = axes[1, 0].imshow(agreement_matrix, cmap="Blues", vmin=0, vmax=1)
        axes[1, 0].set_title("Cluster Agreement Matrix (ARI)")
        axes[1, 0].set_xticks(range(3))
        axes[1, 0].set_yticks(range(3))
        axes[1, 0].set_xticklabels(approaches)
        axes[1, 0].set_yticklabels(approaches)

        # Add text annotations
        for i in range(3):
            for j in range(3):
                axes[1, 0].text(
                    j,
                    i,
                    f"{agreement_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

        plt.colorbar(im, ax=axes[1, 0])

        # 5. PCA Explained Variance
        pca_vars = self.clustering_comparison["pca_info"]["explained_variance_ratio"]
        axes[1, 1].bar(range(1, len(pca_vars) + 1), pca_vars, color="green", alpha=0.8)
        axes[1, 1].set_title(f"PCA Explained Variance\n(Total: {sum(pca_vars):.3f})")
        axes[1, 1].set_xlabel("Principal Component")
        axes[1, 1].set_ylabel("Explained Variance Ratio")
        axes[1, 1].grid(True, alpha=0.3)

        # Add cumulative line
        cumulative = np.cumsum(pca_vars)
        ax_twin = axes[1, 1].twinx()
        ax_twin.plot(range(1, len(pca_vars) + 1), cumulative, "ro-", alpha=0.7)
        ax_twin.set_ylabel("Cumulative Variance", color="red")

        # 6. Summary and Ranking
        ranking = self.clustering_comparison["ranking"]
        summary_text = f"""Clustering Quality Ranking

🥇 1st Place: {ranking['first']}
🥈 2nd Place: {ranking['second']}
🥉 3rd Place: {ranking['third']}

Detailed Scores:
{ranking['first']}: {ranking['scores'][ranking['first']]:.3f}
{ranking['second']}: {ranking['scores'][ranking['second']]:.3f}
{ranking['third']}: {ranking['scores'][ranking['third']]:.3f}

Permutation Test Results (n=1000):
Null: Method has no effect on quality

P-values (α=0.05):
Euc vs Geo: {self.clustering_comparison['statistical_tests']['euclidean_vs_geodesic']['p_value']:.4f} {'***' if self.clustering_comparison['statistical_tests']['euclidean_vs_geodesic']['p_value'] < 0.001 else '**' if self.clustering_comparison['statistical_tests']['euclidean_vs_geodesic']['p_value'] < 0.01 else '*' if self.clustering_comparison['statistical_tests']['euclidean_vs_geodesic']['significant_at_0.05'] else 'ns'}
Euc vs PCA: {self.clustering_comparison['statistical_tests']['euclidean_vs_pca']['p_value']:.4f} {'***' if self.clustering_comparison['statistical_tests']['euclidean_vs_pca']['p_value'] < 0.001 else '**' if self.clustering_comparison['statistical_tests']['euclidean_vs_pca']['p_value'] < 0.01 else '*' if self.clustering_comparison['statistical_tests']['euclidean_vs_pca']['significant_at_0.05'] else 'ns'}
Geo vs PCA: {self.clustering_comparison['statistical_tests']['geodesic_vs_pca']['p_value']:.4f} {'***' if self.clustering_comparison['statistical_tests']['geodesic_vs_pca']['p_value'] < 0.001 else '**' if self.clustering_comparison['statistical_tests']['geodesic_vs_pca']['p_value'] < 0.01 else '*' if self.clustering_comparison['statistical_tests']['geodesic_vs_pca']['significant_at_0.05'] else 'ns'}

Bonferroni-corrected:
α = {self.clustering_comparison['statistical_tests']['bonferroni_alpha']:.4f}
Significant: {'Yes' if any([self.clustering_comparison['statistical_tests']['euclidean_vs_geodesic']['significant_bonferroni'], self.clustering_comparison['statistical_tests']['euclidean_vs_pca']['significant_bonferroni'], self.clustering_comparison['statistical_tests']['geodesic_vs_pca']['significant_bonferroni']]) else 'No'}

PCA Info:
Dims: {self.clustering_comparison['pca_info']['n_components']}
Variance: {self.clustering_comparison['pca_info']['total_variance_explained']:.3f}
"""

        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.1,
            0.9,
            summary_text,
            transform=axes[1, 2].transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "clustering_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("   ✅ Saved comprehensive clustering_comparison.png")

        # Create permutation test visualization
        self._create_permutation_test_plot()

    def _create_permutation_test_plot(self):
        """Create visualization of permutation test results showing null distributions."""
        print("   📊 Creating permutation test visualization...")

        if (
            not hasattr(self, "clustering_comparison")
            or "statistical_tests" not in self.clustering_comparison
        ):
            print("   ⚠️  No permutation test results to visualize")
            return

        perm_results = self.clustering_comparison["statistical_tests"][
            "permutation_results"
        ]

        # Create figure with histograms of permutation distributions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        comparisons = [
            ("euclidean_vs_geodesic", "Euclidean vs Geodesic", axes[0]),
            ("euclidean_vs_pca", "Euclidean vs PCA", axes[1]),
            ("geodesic_vs_pca", "Geodesic vs PCA", axes[2]),
        ]

        for comp_key, title, ax in comparisons:
            # Get permutation distribution and observed value
            perm_diffs = perm_results["permutation_distributions"][comp_key]
            obs_diff = perm_results["observed_differences"][comp_key]
            p_value = self.clustering_comparison["statistical_tests"][comp_key][
                "p_value"
            ]

            if len(perm_diffs) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No valid\npermutations",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(f"{title}\nNo data")
                continue

            # Create histogram of null distribution
            ax.hist(
                perm_diffs,
                bins=50,
                alpha=0.7,
                color="lightblue",
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )

            # Add observed value as vertical line
            ax.axvline(
                obs_diff,
                color="red",
                linewidth=3,
                alpha=0.8,
                label=f"Observed: {obs_diff:.4f}",
            )

            # Add critical values (5% most extreme)
            perm_abs = np.abs(perm_diffs)
            critical_value = np.percentile(perm_abs, 95)
            ax.axvline(
                critical_value,
                color="orange",
                linewidth=2,
                linestyle="--",
                alpha=0.7,
                label=f"95th %ile: ±{critical_value:.4f}",
            )
            ax.axvline(
                -critical_value, color="orange", linewidth=2, linestyle="--", alpha=0.7
            )

            # Shade extreme regions
            extreme_pos = np.array(perm_diffs)[np.array(perm_diffs) >= critical_value]
            extreme_neg = np.array(perm_diffs)[np.array(perm_diffs) <= -critical_value]

            if len(extreme_pos) > 0:
                ax.hist(extreme_pos, bins=20, alpha=0.3, color="red", density=True)
            if len(extreme_neg) > 0:
                ax.hist(extreme_neg, bins=20, alpha=0.3, color="red", density=True)

            # Labels and title
            ax.set_xlabel("Silhouette Score Difference")
            ax.set_ylabel("Density")
            ax.set_title(
                f"{title}\np = {p_value:.4f} "
                + (
                    "***"
                    if p_value < 0.001
                    else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "(ns)"
                )
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add interpretation text
            interpretation = self.clustering_comparison["statistical_tests"][comp_key][
                "interpretation"
            ]
            significance = "Significant" if p_value < 0.05 else "Not significant"
            ax.text(
                0.02,
                0.98,
                f"{interpretation}\n({significance})",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10,
            )

        plt.suptitle(
            "Permutation Test Results: Null Distributions vs Observed Differences",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "permutation_test_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("   ✅ Saved permutation_test_results.png")

    def create_2d_cluster_maps(self):
        """Create comprehensive 2D cluster maps for all three clustering approaches."""
        print("\n📊 Creating 2D cluster maps for all clustering approaches...")

        if not hasattr(self, "labels_euclidean"):
            self.perform_clustering()

        # Create figure with subplots: 2 rows x 3 cols
        # Top row: colored by cluster, Bottom row: colored by time
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Define color maps
        cluster_cmap = "Set1"
        time_cmap = "viridis"

        # Plot settings for each approach
        plot_configs = [
            {
                "coords": self.z_euclidean_mds,
                "labels": self.labels_euclidean,
                "title_base": "Euclidean Clustering",
                "xlabel": "MDS Euclidean Dim 1",
                "ylabel": "MDS Euclidean Dim 2",
                "col": 0,
            },
            {
                "coords": self.z_geodesic_2d,
                "labels": self.labels_geodesic,
                "title_base": "Geodesic Clustering",
                "xlabel": "MDS Geodesic Dim 1",
                "ylabel": "MDS Geodesic Dim 2",
                "col": 1,
            },
            {
                "coords": self.z_pca[:, :2],  # Use first 2 PCA components
                "labels": self.labels_pca,
                "title_base": "PCA Clustering",
                "xlabel": "PCA Component 1",
                "ylabel": "PCA Component 2",
                "col": 2,
            },
        ]

        # Row 1: Colored by cluster
        for config in plot_configs:
            ax = axes[0, config["col"]]

            # Create scatter plot colored by cluster
            scatter = ax.scatter(
                config["coords"][:, 0],
                config["coords"][:, 1],
                c=config["labels"],
                cmap=cluster_cmap,
                s=15,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.2,
            )

            ax.set_title(
                f"{config['title_base']}\n(Colored by Cluster)",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel(config["xlabel"])
            ax.set_ylabel(config["ylabel"])
            ax.grid(True, alpha=0.3)

            # Add colorbar for clusters
            n_clusters = len(np.unique(config["labels"]))
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Cluster ID")
            cbar.set_ticks(range(n_clusters))

        # Row 2: Colored by time
        for config in plot_configs:
            ax = axes[1, config["col"]]

            # Create scatter plot colored by time
            scatter = ax.scatter(
                config["coords"][:, 0],
                config["coords"][:, 1],
                c=self.time_indices,
                cmap=time_cmap,
                s=15,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.2,
            )

            ax.set_title(
                f"{config['title_base']}\n(Colored by Time)",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel(config["xlabel"])
            ax.set_ylabel(config["ylabel"])
            ax.grid(True, alpha=0.3)

            # Add colorbar for time
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Time Index")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "2d_cluster_maps.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("   ✅ Saved 2d_cluster_maps.png")

        # Create individual detailed plots for each approach
        self._create_individual_cluster_plots()

    def _create_individual_cluster_plots(self):
        """Create individual detailed plots for each clustering approach."""
        print("   📊 Creating individual detailed cluster plots...")

        plot_configs = [
            {
                "coords": self.z_euclidean_mds,
                "labels": self.labels_euclidean,
                "approach": "euclidean",
                "title": "Euclidean Clustering (MDS of Euclidean Distances)",
                "xlabel": "MDS Euclidean Dimension 1",
                "ylabel": "MDS Euclidean Dimension 2",
            },
            {
                "coords": self.z_geodesic_2d,
                "labels": self.labels_geodesic,
                "approach": "geodesic",
                "title": "Geodesic Clustering (MDS of Geodesic Distances)",
                "xlabel": "MDS Geodesic Dimension 1",
                "ylabel": "MDS Geodesic Dimension 2",
            },
            {
                "coords": self.z_pca[:, :2],
                "labels": self.labels_pca,
                "approach": "pca",
                "title": "PCA Clustering (First 2 Principal Components)",
                "xlabel": f"PC1 (Var: {self.pca.explained_variance_ratio_[0]:.3f})",
                "ylabel": f"PC2 (Var: {self.pca.explained_variance_ratio_[1]:.3f})",
            },
        ]

        for config in plot_configs:
            # Create figure with subplots for cluster and time coloring
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Left plot: colored by cluster
            unique_clusters = np.unique(config["labels"])
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

            for i, cluster_id in enumerate(unique_clusters):
                mask = config["labels"] == cluster_id
                ax1.scatter(
                    config["coords"][mask, 0],
                    config["coords"][mask, 1],
                    c=[colors[i]],
                    label=f"Cluster {cluster_id}",
                    s=20,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.3,
                )

            ax1.set_title(
                f"{config['title']}\n(Colored by Cluster)",
                fontsize=14,
                fontweight="bold",
            )
            ax1.set_xlabel(config["xlabel"])
            ax1.set_ylabel(config["ylabel"])
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Right plot: colored by time with trajectory lines
            # Sort points by time for trajectory
            time_order = np.argsort(self.time_indices)
            coords_ordered = config["coords"][time_order]
            time_ordered = self.time_indices[time_order]

            # Plot trajectory line
            ax2.plot(
                coords_ordered[:, 0],
                coords_ordered[:, 1],
                color="gray",
                alpha=0.3,
                linewidth=0.5,
                zorder=1,
            )

            # Plot points colored by time
            scatter = ax2.scatter(
                config["coords"][:, 0],
                config["coords"][:, 1],
                c=self.time_indices,
                cmap="viridis",
                s=20,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.3,
                zorder=2,
            )

            ax2.set_title(
                f"{config['title']}\n(Colored by Time with Trajectory)",
                fontsize=14,
                fontweight="bold",
            )
            ax2.set_xlabel(config["xlabel"])
            ax2.set_ylabel(config["ylabel"])
            ax2.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label("Time Index")

            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"2d_{config['approach']}_clustering_detailed.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        print("   ✅ Saved individual detailed cluster plots")

        # Create comparison plot showing cluster boundaries
        self._create_cluster_boundary_comparison()

    def _create_cluster_boundary_comparison(self):
        """Create a comparison plot showing decision boundaries for each clustering approach."""
        print("   📊 Creating cluster boundary comparison...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        plot_configs = [
            {
                "coords": self.z_euclidean_mds,
                "labels": self.labels_euclidean,
                "title": "Euclidean\n(MDS Euclidean)",
                "ax": axes[0],
            },
            {
                "coords": self.z_geodesic_2d,
                "labels": self.labels_geodesic,
                "title": "Geodesic\n(MDS Geodesic)",
                "ax": axes[1],
            },
            {
                "coords": self.z_pca[:, :2],
                "labels": self.labels_pca,
                "title": "PCA\n(PC1 vs PC2)",
                "ax": axes[2],
            },
        ]

        for config in plot_configs:
            ax = config["ax"]
            coords = config["coords"]
            labels = config["labels"]

            # Create a mesh for decision boundary visualization
            h = 0.02  # Step size in the mesh
            x_min, x_max = coords[:, 0].min() - 1, coords[:, 0].max() + 1
            y_min, y_max = coords[:, 1].min() - 1, coords[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            # Train a simple classifier to show decision boundaries
            from sklearn.neighbors import KNeighborsClassifier

            clf = KNeighborsClassifier(n_neighbors=15)
            clf.fit(coords, labels)

            # Predict on mesh
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.3, cmap="Set1")

            # Plot points
            unique_clusters = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

            for i, cluster_id in enumerate(unique_clusters):
                mask = labels == cluster_id
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=[colors[i]],
                    s=30,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.5,
                    label=f"Cluster {cluster_id}",
                )

            ax.set_title(config["title"], fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.suptitle(
            "Clustering Decision Boundaries Comparison", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "2d_cluster_boundaries_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("   ✅ Saved cluster boundaries comparison")

    def create_3d_geodesic_plot(self):
        """Create a 3D matplotlib plot of geodesic clustering using top 3 MDS dimensions."""
        print("\n🎨 Creating 3D geodesic clustering plot...")

        if not hasattr(self, "labels_geodesic"):
            self.perform_clustering()

        # Create figure with 3D subplot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Get unique clusters and create colors
        unique_clusters = np.unique(self.labels_geodesic)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            mask = self.labels_geodesic == cluster_id
            cluster_points = self.z_geodesic_3d[mask]

            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                cluster_points[:, 2],
                c=[colors[i]],
                s=30,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.3,
                label=f"Cluster {cluster_id}",
            )

        # Customize the plot
        ax.set_xlabel("MDS Geodesic Dimension 1")
        ax.set_ylabel("MDS Geodesic Dimension 2")
        ax.set_zlabel("MDS Geodesic Dimension 3")
        ax.set_title(
            "3D Geodesic K-Means Clustering\n(MDS of Geodesic Distances)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "3d_geodesic_clustering.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("   ✅ Saved 3d_geodesic_clustering.png")

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
            "PCA_Cluster": self.labels_pca,
        }
        if self.has_dates:
            summary_data["Date"] = self.dates.strftime("%Y-%m-%d")

        # Add PCA components if they exist
        if hasattr(self, "z_pca"):
            for i in range(self.pca_dims):
                summary_data[f"PCA_{i+1}"] = self.z_pca[:, i]

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
            "pca_dimensions": int(self.pca_dims),
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
            ranking = self.clustering_comparison["ranking"]
            pca_var = self.clustering_comparison["pca_info"]["total_variance_explained"]
            print(
                f"   🏆 Clustering ranking: 1st={ranking['first']}, 2nd={ranking['second']}, 3rd={ranking['third']}"
            )
            print(f"   📐 PCA variance explained: {pca_var:.3f}")

    def run_full_analysis(self, n_clusters=5):
        """Run the complete latent geometry analysis with three clustering approaches."""
        print(
            "\n🚀 Running full latent geometry analysis with three clustering approaches..."
        )

        # Step 1: Compute geometric properties
        self.compute_jacobians()
        self.compute_riemannian_metrics()
        self.compute_geodesic_distances()

        # Step 2: Perform clustering (now includes PCA clustering)
        self.perform_clustering(n_clusters=n_clusters)

        # Step 3: Create 2D cluster visualizations
        self.create_2d_cluster_maps()

        # Step 4: Create 3D geodesic visualization
        self.create_3d_geodesic_plot()

        # Step 5: Save summary
        self.save_summary_data()

        print(f"\n🎉 Latent geometry analysis complete!")
        print(f"📁 All results saved to: {self.output_dir}")

        print(f"📊 2D Cluster visualizations:")
        print(f"   - {self.output_dir}/2d_cluster_maps.png")
        print(f"   - {self.output_dir}/2d_euclidean_clustering_detailed.png")
        print(f"   - {self.output_dir}/2d_geodesic_clustering_detailed.png")
        print(f"   - {self.output_dir}/2d_pca_clustering_detailed.png")
        print(f"   - {self.output_dir}/2d_cluster_boundaries_comparison.png")
        print(f"🎨 3D Visualization:")
        print(f"   - {self.output_dir}/3d_geodesic_clustering.png")
        print(f"📊 Clustering analyses:")
        print(f"   - {self.output_dir}/clustering_comparison.png")
        print(f"   - {self.output_dir}/clustering_comparison.json")
        print(f"   - {self.output_dir}/permutation_test_results.png")
        print(
            f"📈 Three clustering approaches compared: Euclidean, Geodesic (MDS), PCA"
        )


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
