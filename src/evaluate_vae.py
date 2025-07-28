"""
Simple VAE Evaluation Script

This script performs essential evaluation of a trained VAE model:
- Latent space correlation analysis (orthogonality)
- Training/validation loss curves
- Distribution comparison (train/val/test vs N(0,1))
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Optional
import warnings
from scipy import stats
import argparse

# Suppress warnings
warnings.filterwarnings("ignore")

# Set style
plt.style.use("default")
sns.set_palette("husl")


class SimpleVAEEvaluator:
    """Simple VAE evaluation focused on essential metrics."""

    def __init__(self, results_path: str, output_dir: Optional[str] = None):
        """
        Initialize evaluator with trained model results.

        Args:
            results_path: Path to training results folder
            output_dir: Output directory for evaluation results
        """
        self.results_path = Path(results_path)
        self.output_dir = (
            Path(output_dir) if output_dir else self.results_path / "evaluation"
        )
        self.output_dir.mkdir(exist_ok=True)

        # Load all data and metadata
        self._load_data()

        print(f"📊 Simple VAE Evaluator initialized")
        print(f"🔍 Results path: {self.results_path}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🧠 Latent dimensions: {self.latent_dim}")

    def _load_data(self):
        """Load latent representations, training data, and metadata."""
        try:
            # Load latent representations
            self.train_latent = np.load(self.results_path / "train_latent.npy")
            self.val_latent = np.load(self.results_path / "val_latent.npy")
            self.test_latent = np.load(self.results_path / "test_latent.npy")

            # Load configuration and metadata
            with open(self.results_path / "training_config.json", "r") as f:
                self.config = json.load(f)

            # Load training history
            self.history = pd.read_csv(self.results_path / "training_history.csv")

            # Extract dimensions
            self.latent_dim = self.config["model_config"]["latent_dim"]
            self.input_dim = self.config["model_config"]["input_dim"]

            print(f"✅ Data loaded successfully")
            print(
                f"📊 Shapes - Train: {self.train_latent.shape}, Val: {self.val_latent.shape}, Test: {self.test_latent.shape}"
            )

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def analyze_correlations(self):
        """Analyze latent space correlations (orthogonality)."""
        print("\n🔍 Analyzing latent space correlations...")

        # Combine all latent vectors
        all_latent = np.vstack([self.train_latent, self.val_latent, self.test_latent])

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(all_latent.T)

        # Calculate off-diagonal correlations (measure of non-orthogonality)
        off_diag_mask = ~np.eye(self.latent_dim, dtype=bool)
        off_diag_corrs = corr_matrix[off_diag_mask]

        # Statistics
        mean_abs_corr = np.mean(np.abs(off_diag_corrs))
        max_abs_corr = np.max(np.abs(off_diag_corrs))

        # Create correlation plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Correlation heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            ax=axes[0],
            fmt=".2f",
            cbar_kws={"label": "Correlation"},
        )
        axes[0].set_title("Latent Dimensions Correlation Matrix")
        axes[0].set_xlabel("Latent Dimension")
        axes[0].set_ylabel("Latent Dimension")

        # Off-diagonal correlations distribution
        axes[1].hist(off_diag_corrs, bins=30, alpha=0.7, edgecolor="black")
        axes[1].axvline(
            0, color="red", linestyle="--", alpha=0.7, label="Perfect Orthogonality"
        )
        axes[1].set_xlabel("Correlation Coefficient")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Off-Diagonal Correlations")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"Mean |correlation|: {mean_abs_corr:.3f}\nMax |correlation|: {max_abs_corr:.3f}"
        axes[1].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "correlation_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Save metrics
        correlation_metrics = {
            "mean_abs_correlation": float(mean_abs_corr),
            "max_abs_correlation": float(max_abs_corr),
            "correlation_matrix": corr_matrix.tolist(),
        }

        with open(self.output_dir / "correlation_metrics.json", "w") as f:
            json.dump(correlation_metrics, f, indent=2)

        print(f"✅ Correlation analysis complete")
        print(f"   📊 Mean |correlation|: {mean_abs_corr:.4f}")
        print(f"   📊 Max |correlation|: {max_abs_corr:.4f}")

        return correlation_metrics

    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        print("\n📈 Plotting training curves...")

        # Create training history plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Total loss
        axes[0].plot(self.history["train_loss"], label="Train", alpha=0.8, linewidth=2)
        axes[0].plot(
            self.history["val_loss"], label="Validation", alpha=0.8, linewidth=2
        )
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Reconstruction loss (most relevant for Beta=0)
        axes[1].plot(self.history["train_recon"], label="Train", alpha=0.8, linewidth=2)
        axes[1].plot(
            self.history["val_recon"], label="Validation", alpha=0.8, linewidth=2
        )
        axes[1].set_title("Reconstruction Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Add final loss values as text
        final_train_loss = self.history["train_loss"].iloc[-1]
        final_val_loss = self.history["val_loss"].iloc[-1]
        best_val_loss = self.config["training_results"]["best_val_loss"]

        loss_text = f"Final Train: {final_train_loss:.6f}\nFinal Val: {final_val_loss:.6f}\nBest Val: {best_val_loss:.6f}"
        axes[0].text(
            0.02,
            0.98,
            loss_text,
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"✅ Training curves saved")
        print(f"   📊 Best validation loss: {best_val_loss:.6f}")
        print(
            f"   📊 Final train/val loss: {final_train_loss:.6f}/{final_val_loss:.6f}"
        )

    def analyze_distributions(self):
        """Analyze latent variable distributions across train/val/test."""
        print("\n📊 Analyzing latent distributions...")

        datasets = {
            "Train": self.train_latent,
            "Validation": self.val_latent,
            "Test": self.test_latent,
        }

        # Calculate grid layout for all dimensions
        if self.latent_dim <= 6:
            nrows, ncols = 2, 3
        elif self.latent_dim <= 12:
            nrows, ncols = 3, 4
        elif self.latent_dim <= 16:
            nrows, ncols = 4, 4
        else:
            nrows, ncols = 4, 5

        # Distribution comparison plots
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        fig.suptitle(
            f"Latent Dimensions Distribution (All {self.latent_dim} Dimensions)",
            fontsize=16,
        )
        axes = axes.flatten()

        # Calculate statistics for summary
        distribution_stats = {}

        for dim in range(self.latent_dim):
            ax = axes[dim]

            # Plot histograms for all datasets
            colors = ["blue", "orange", "green"]
            for i, (dataset_name, data) in enumerate(datasets.items()):
                dim_data = data[:, dim]
                ax.hist(
                    dim_data,
                    bins=50,
                    alpha=0.6,
                    density=True,
                    label=dataset_name,
                    color=colors[i],
                    histtype="stepfilled",
                )

                # Store statistics
                if dataset_name not in distribution_stats:
                    distribution_stats[dataset_name] = {}
                distribution_stats[dataset_name][f"dim_{dim+1}"] = {
                    "mean": float(np.mean(dim_data)),
                    "std": float(np.std(dim_data)),
                }

            # Add N(0,1) reference
            x_range = np.linspace(-4, 4, 100)
            ax.plot(
                x_range,
                stats.norm.pdf(x_range, 0, 1),
                "k--",
                alpha=0.8,
                linewidth=2,
                label="N(0,1)",
            )

            ax.set_title(f"Latent Dimension {dim+1}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            if dim == 0:  # Only show legend on first subplot
                ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for dim in range(self.latent_dim, len(axes)):
            axes[dim].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "distribution_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Create summary statistics table
        stats_summary = []
        for dataset_name in datasets.keys():
            for dim in range(self.latent_dim):
                stats_data = distribution_stats[dataset_name][f"dim_{dim+1}"]
                stats_summary.append(
                    {
                        "Dataset": dataset_name,
                        "Dimension": dim + 1,
                        "Mean": stats_data["mean"],
                        "Std": stats_data["std"],
                    }
                )

        stats_df = pd.DataFrame(stats_summary)
        stats_df.to_csv(self.output_dir / "distribution_statistics.csv", index=False)

        # Save detailed statistics
        with open(self.output_dir / "distribution_stats.json", "w") as f:
            json.dump(distribution_stats, f, indent=2)

        print(f"✅ Distribution analysis complete")

        # Print summary of how close distributions are to N(0,1)
        train_means = [
            distribution_stats["Train"][f"dim_{dim+1}"]["mean"]
            for dim in range(self.latent_dim)
        ]
        train_stds = [
            distribution_stats["Train"][f"dim_{dim+1}"]["std"]
            for dim in range(self.latent_dim)
        ]

        mean_dev = np.mean(np.abs(train_means))
        std_dev = np.mean(np.abs(np.array(train_stds) - 1.0))

        print(f"   📊 Average |mean| deviation from 0: {mean_dev:.4f}")
        print(f"   📊 Average |std| deviation from 1: {std_dev:.4f}")

        return distribution_stats

    def generate_summary_report(self):
        """Generate a simple evaluation summary."""
        print("\n📋 Generating evaluation summary...")

        # Run all analyses
        correlation_metrics = self.analyze_correlations()
        self.plot_training_curves()
        distribution_stats = self.analyze_distributions()

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create summary report
        report = f"""# VAE Evaluation Summary

**Generated on:** {timestamp}  
**Model Path:** {self.results_path}  
**Evaluation Output:** {self.output_dir}

## Model Configuration

- **Input Dimension:** {self.input_dim}
- **Latent Dimension:** {self.latent_dim}
- **Hidden Dimension:** {self.config['model_config']['hidden_dim']}
- **Training Loss (Best):** {self.config['training_results']['best_val_loss']:.6f}
- **Training Time:** {self.config['training_results']['training_time_minutes']:.1f} minutes

## Dataset Information

- **Training Samples:** {len(self.train_latent):,}
- **Validation Samples:** {len(self.val_latent):,}
- **Test Samples:** {len(self.test_latent):,}

## Results Summary

### 1. Latent Space Orthogonality
- **Mean |Correlation|:** {correlation_metrics['mean_abs_correlation']:.4f}
- **Max |Correlation|:** {correlation_metrics['max_abs_correlation']:.4f}

### 2. Training Performance
- **Best Validation Loss:** {self.config['training_results']['best_val_loss']:.6f}
- **Final Training Loss:** {self.history['train_loss'].iloc[-1]:.6f}
- **Final Validation Loss:** {self.history['val_loss'].iloc[-1]:.6f}
- **Total Epochs:** {len(self.history)}

### 3. Distribution Quality
Training set statistics (deviation from N(0,1)):
"""

        # Add distribution summary
        train_means = [
            distribution_stats["Train"][f"dim_{dim+1}"]["mean"]
            for dim in range(self.latent_dim)
        ]
        train_stds = [
            distribution_stats["Train"][f"dim_{dim+1}"]["std"]
            for dim in range(self.latent_dim)
        ]

        mean_dev = np.mean(np.abs(train_means))
        std_dev = np.mean(np.abs(np.array(train_stds) - 1.0))

        report += f"""
- **Average |Mean| Deviation:** {mean_dev:.4f}
- **Average |Std| Deviation:** {std_dev:.4f}

## Generated Files

- `correlation_analysis.png` - Correlation matrix and distribution
- `training_curves.png` - Training and validation loss curves  
- `distribution_analysis.png` - Latent distributions vs N(0,1)
- `correlation_metrics.json` - Correlation statistics
- `distribution_statistics.csv` - Summary statistics table
- `distribution_stats.json` - Detailed distribution statistics

## Quick Assessment

"""

        # Add quick assessment
        assessments = []

        if correlation_metrics["mean_abs_correlation"] < 0.1:
            assessments.append(
                "✅ **Excellent orthogonality** - latent dimensions are well-separated"
            )
        elif correlation_metrics["mean_abs_correlation"] < 0.3:
            assessments.append(
                "🟡 **Good orthogonality** - some correlation between dimensions"
            )
        else:
            assessments.append(
                "❌ **Poor orthogonality** - high correlation between dimensions"
            )

        if mean_dev < 0.1 and std_dev < 0.2:
            assessments.append("✅ **Good distribution match** - close to N(0,1)")
        elif mean_dev < 0.3 and std_dev < 0.5:
            assessments.append(
                "🟡 **Reasonable distribution** - some deviation from N(0,1)"
            )
        else:
            assessments.append(
                "❌ **Poor distribution** - significant deviation from N(0,1)"
            )

        for assessment in assessments:
            report += f"{assessment}\n\n"

        report += "---\n*Simple VAE Evaluator*"

        # Save report
        with open(self.output_dir / "EVALUATION_SUMMARY.md", "w") as f:
            f.write(report)

        print(f"✅ Evaluation complete!")
        print(f"📋 Summary saved to: {self.output_dir}/EVALUATION_SUMMARY.md")
        print(f"📁 All files saved to: {self.output_dir}")

        return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Simple VAE Evaluation")
    parser.add_argument("results_path", help="Path to VAE training results folder")
    parser.add_argument(
        "--output_dir", help="Output directory for evaluation results", default=None
    )

    args = parser.parse_args()

    # Create evaluator and run analysis
    evaluator = SimpleVAEEvaluator(args.results_path, args.output_dir)
    evaluator.generate_summary_report()


if __name__ == "__main__":
    main()
