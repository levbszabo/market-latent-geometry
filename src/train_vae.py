# Core imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleVAE, vae_loss_fn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import os
import pickle


# Setup
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("✅ All packages imported successfully!")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"💻 Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎯 Using device: {device}")

# Create timestamped results folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("../results") / f"vae_training_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)
print(f"📁 Results will be saved to: {results_dir}")

from config import (
    HIDDEN_DIM,
    LATENT_DIM,
    DROPOUT_RATE,
    BETA,
    C,
    LAMBDA_ORTHO,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    WEIGHT_DECAY,
)

# Set data path
DATA_PATH = "../processed_data_simple/latest"
data_path = Path(DATA_PATH)

print(f"📁 Loading data from: {data_path}")
print(f"📂 Path exists: {data_path.exists()}")

# Load the preprocessed data
try:
    # Load VAE-ready numpy arrays (already normalized)
    X_train = np.load(data_path / "train_vae.npy")
    X_val = np.load(data_path / "validation_vae.npy")
    X_test = np.load(data_path / "test_vae.npy")

    # Load metadata
    with open(data_path / "dataset_metadata.json", "r") as f:
        metadata = json.load(f)

    # Load normalization stats (for reference)
    norm_stats = pd.read_csv(data_path / "normalization_stats.csv", index_col=0)

    print(f"✅ Data loaded successfully!")
    print(f"📊 Training set shape: {X_train.shape}")
    print(f"📊 Validation set shape: {X_val.shape}")
    print(f"📊 Test set shape: {X_test.shape}")
    print(f"🎯 Input dimension: {X_train.shape[1]}")
    print(f"🏢 Number of tickers: {metadata['n_tickers']}")
    print(f"📈 Features per ticker: {metadata['n_features_per_ticker']}")

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    # Create dates for visualization (use metadata if available)
    if "train_dates" in metadata:
        train_dates = pd.to_datetime(metadata["train_dates"])
        val_dates = pd.to_datetime(metadata["val_dates"])
        test_dates = pd.to_datetime(metadata["test_dates"])
    else:
        # Create default date ranges
        print("📅 Creating default date ranges...")
        total_days = len(X_train) + len(X_val) + len(X_test)
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=total_days - 1)
        all_dates = pd.date_range(start=start_date, periods=total_days, freq="D")

        train_dates = all_dates[: len(X_train)]
        val_dates = all_dates[len(X_train) : len(X_train) + len(X_val)]
        test_dates = all_dates[len(X_train) + len(X_val) :]

    print(f"📅 Date ranges:")
    print(
        f"   Train: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}"
    )
    print(
        f"   Val: {val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')}"
    )
    print(
        f"   Test: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}"
    )

    # Show normalization stats
    print(f"\n📊 Normalization Statistics:")
    print(norm_stats)

except Exception as e:
    print(f"❌ Error loading data: {e}")
    print(f"Please ensure the following files exist in {data_path}:")
    print("  - train_vae.npy")
    print("  - validation_vae.npy")
    print("  - test_vae.npy")
    print("  - dataset_metadata.json")
    print("  - normalization_stats.csv")


if "X_train" in locals():
    # INPUT_DIM is automatically calculated from the loaded data
    INPUT_DIM = X_train.shape[1]

    # All other dimensions come from the hyperparameter configuration section
    # HIDDEN_DIM, LATENT_DIM, DROPOUT_RATE are already set above

    print(f"🎯 Model Configuration Summary:")
    print(f"   📊 Input dimension: {INPUT_DIM} (calculated from data)")
    print(f"   🏗️ Hidden dimension: {HIDDEN_DIM} (from config)")
    print(f"   🧠 Latent dimension: {LATENT_DIM} (from config)")
    print(f"   🛡️ Dropout rate: {DROPOUT_RATE} (from config)")
    print(f"   📏 Compression ratio: {INPUT_DIM/LATENT_DIM:.1f}:1")

    # Validation checks
    if HIDDEN_DIM < 8 * LATENT_DIM:
        print(
            f"   ⚠️  Warning: HIDDEN_DIM ({HIDDEN_DIM}) < 8 × LATENT_DIM ({8 * LATENT_DIM})"
        )
        print(
            f"      Consider increasing HIDDEN_DIM to at least {8 * LATENT_DIM} for better capacity"
        )

    if LATENT_DIM > 32:
        print(f"   ⚠️  Warning: LATENT_DIM ({LATENT_DIM}) is quite high (>32)")
        print(f"      Consider reducing to 8-16 for more interpretable factors")

else:
    print("❌ Data not loaded - cannot configure model")
    print("💡 Please run the data loading section first")

if "INPUT_DIM" in locals():
    print("🏗️ Creating model with configured hyperparameters...")

    # Create model using all hyperparameters from config section
    model = SimpleVAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🧠 Model Summary:")
    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    print(f"   💾 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # Create optimizer using configured hyperparameters
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Create data loaders using configured batch size
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n🚀 Training Configuration Summary:")
    print(f"   📦 Batch size: {BATCH_SIZE} (from config)")
    print(f"   📈 Learning rate: {LEARNING_RATE} (from config)")
    print(f"   🛡️ Weight decay: {WEIGHT_DECAY} (from config)")
    print(f"   🔄 Max epochs: {NUM_EPOCHS} (from config)")
    print(f"   ⏱️ Patience: {PATIENCE} (from config)")

    print(f"\n📊 VAE Loss Configuration:")
    print(f"   🔀 β (KL weight): {BETA} (from config)")
    print(f"   🎯 C (KL capacity): {C} (from config)")
    print(f"   🔗 λ (ortho penalty): {LAMBDA_ORTHO} (from config)")

    print(f"\n📚 Data Loader Summary:")
    print(f"   🏋️ Train batches: {len(train_loader)}")
    print(f"   🧪 Validation batches: {len(val_loader)}")
    print(f"   📊 Samples per batch: {BATCH_SIZE}")

    # Configuration validation
    if PATIENCE >= NUM_EPOCHS:
        print(f"\n⚠️  Warning: PATIENCE ({PATIENCE}) ≥ NUM_EPOCHS ({NUM_EPOCHS})")
        print(
            f"   Early stopping will never trigger. Consider reducing PATIENCE to ~{NUM_EPOCHS//5}"
        )

    if LEARNING_RATE > 1e-3:
        print(
            f"\n⚠️  Warning: LEARNING_RATE ({LEARNING_RATE}) is quite high for financial data"
        )
        print(f"   Consider reducing to ≤1e-4 for more stable training")

    print(f"\n✅ Training setup complete - ready to train!")

else:
    print("❌ Cannot create model - INPUT_DIM not available")
    print("💡 Please run the data loading and model configuration sections first")


# Training loop with early stopping
if "model" in locals():
    print("🏋️ Starting training...")
    start_time = time.time()

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_recon": [],
        "val_kl": [],
        "train_ortho": [],
        "val_ortho": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    try:
        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()

            # Training phase
            model.train()
            train_loss = 0
            train_recon = 0
            train_kl = 0
            train_ortho = 0

            for batch_idx, (data,) in enumerate(train_loader):
                optimizer.zero_grad()

                # Forward pass
                x_recon, mean, log_var, z = model(data)
                # Calculate loss using hyperparameters
                total_loss, recon_loss, kl_loss, ortho_penalty = vae_loss_fn(
                    x_recon, data, mean, log_var, BETA, C, LAMBDA_ORTHO
                )

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Accumulate losses
                train_loss += total_loss.item()
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()
                train_ortho += ortho_penalty.item()
            # Average training losses
            train_loss /= len(train_loader)
            train_recon /= len(train_loader)
            train_kl /= len(train_loader)
            train_ortho /= len(train_loader)
            # Validation phase
            model.eval()
            val_loss = 0
            val_recon = 0
            val_kl = 0
            val_ortho = 0
            with torch.no_grad():
                for (data,) in val_loader:
                    x_recon, mean, log_var, z = model(data)
                    total_loss, recon_loss, kl_loss, ortho_penalty = vae_loss_fn(
                        x_recon, data, mean, log_var, BETA, C, LAMBDA_ORTHO
                    )

                    val_loss += total_loss.item()
                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()
                    val_ortho += ortho_penalty.item()
            # Average validation losses
            val_loss /= len(val_loader)
            val_recon /= len(val_loader)
            val_kl /= len(val_loader)
            val_ortho /= len(val_loader)
            # Store history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_recon"].append(train_recon)
            history["train_kl"].append(train_kl)
            history["val_recon"].append(val_recon)
            history["val_kl"].append(val_kl)
            history["train_ortho"].append(train_ortho)
            history["val_ortho"].append(val_ortho)
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # Print progress
            epoch_time = time.time() - epoch_start
            if epoch % 5 == 0 or epoch < 10:
                print(
                    f"Epoch {epoch:3d}/{NUM_EPOCHS} ({epoch_time:.1f}s): "
                    f"Train={train_loss:.4f} (R:{train_recon:.4f}, KL:{train_kl:.4f}, O:{train_ortho:.4f}), "
                    f"Val={val_loss:.4f} (R:{val_recon:.4f}, KL:{val_kl:.4f}, O:{val_ortho:.4f})"
                )

            # Early stopping
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model (val_loss={best_val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time/60:.1f} minutes!")
    print(f"🎯 Best validation loss: {best_val_loss:.4f}")

    # Save all outputs
    print(f"\n💾 Saving training outputs to {results_dir}...")

    # 1. Save model weights
    torch.save(model.state_dict(), results_dir / "model_weights.pth")
    torch.save(model, results_dir / "model_complete.pth")
    print("✅ Model weights saved")

    # 2. Save training configuration and metadata
    training_config = {
        "model_config": {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "dropout_rate": DROPOUT_RATE,
        },
        "training_config": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "weight_decay": WEIGHT_DECAY,
        },
        "loss_config": {"beta": BETA, "c": C, "lambda_ortho": LAMBDA_ORTHO},
        "training_results": {
            "best_val_loss": best_val_loss,
            "total_epochs": len(history["train_loss"]),
            "training_time_minutes": total_time / 60,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "data_info": metadata.copy() if "metadata" in locals() else {},
        "timestamp": timestamp,
    }

    with open(results_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    print("✅ Training configuration saved")

    # 3. Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "training_history.csv", index=False)

    # Save as pickle for easy loading
    with open(results_dir / "training_history.pkl", "wb") as f:
        pickle.dump(history, f)
    print("✅ Training history saved")

    # 4. Generate and save latent representations
    print("🧠 Generating latent representations...")
    model.eval()
    with torch.no_grad():
        # Generate latent vectors for all datasets
        _, train_mean, train_log_var, train_z = model(X_train_tensor)
        _, val_mean, val_log_var, val_z = model(X_val_tensor)
        _, test_mean, test_log_var, test_z = model(X_test_tensor)

        # Convert to numpy
        train_latent = train_z.cpu().numpy()
        val_latent = val_z.cpu().numpy()
        test_latent = test_z.cpu().numpy()

        train_mean_np = train_mean.cpu().numpy()
        val_mean_np = val_mean.cpu().numpy()
        test_mean_np = test_mean.cpu().numpy()

        train_logvar_np = train_log_var.cpu().numpy()
        val_logvar_np = val_log_var.cpu().numpy()
        test_logvar_np = test_log_var.cpu().numpy()

    # Save latent representations
    np.save(results_dir / "train_latent.npy", train_latent)
    np.save(results_dir / "val_latent.npy", val_latent)
    np.save(results_dir / "test_latent.npy", test_latent)

    np.save(results_dir / "train_mean.npy", train_mean_np)
    np.save(results_dir / "val_mean.npy", val_mean_np)
    np.save(results_dir / "test_mean.npy", test_mean_np)

    np.save(results_dir / "train_logvar.npy", train_logvar_np)
    np.save(results_dir / "val_logvar.npy", val_logvar_np)
    np.save(results_dir / "test_logvar.npy", test_logvar_np)
    print("✅ Latent representations saved")

    # 5. Save date information
    if "train_dates" in locals():
        date_info = {
            "train_dates": train_dates.strftime("%Y-%m-%d").tolist(),
            "val_dates": val_dates.strftime("%Y-%m-%d").tolist(),
            "test_dates": test_dates.strftime("%Y-%m-%d").tolist(),
        }
        with open(results_dir / "date_info.json", "w") as f:
            json.dump(date_info, f, indent=2)
        print("✅ Date information saved")

    # 6. Generate and save plots
    print("📊 Generating and saving plots...")

    # Set style for better plots
    plt.style.use("default")
    sns.set_palette("husl")

    # Plot 1: Training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"VAE Training History (Best Val Loss: {best_val_loss:.4f})", fontsize=16
    )

    # Total loss
    axes[0, 0].plot(history["train_loss"], label="Train", alpha=0.8)
    axes[0, 0].plot(history["val_loss"], label="Validation", alpha=0.8)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[0, 1].plot(history["train_recon"], label="Train", alpha=0.8)
    axes[0, 1].plot(history["val_recon"], label="Validation", alpha=0.8)
    axes[0, 1].set_title("Reconstruction Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL divergence
    axes[1, 0].plot(history["train_kl"], label="Train", alpha=0.8)
    axes[1, 0].plot(history["val_kl"], label="Validation", alpha=0.8)
    axes[1, 0].set_title("KL Divergence")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("KL Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Orthogonality penalty
    axes[1, 1].plot(history["train_ortho"], label="Train", alpha=0.8)
    axes[1, 1].plot(history["val_ortho"], label="Validation", alpha=0.8)
    axes[1, 1].set_title("Orthogonality Penalty")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Penalty")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Latent space visualization (first 2 dimensions)
    if LATENT_DIM >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Latent Space Visualization (First 2 Dimensions)", fontsize=16)

        # Training data
        axes[0].scatter(train_latent[:, 0], train_latent[:, 1], alpha=0.6, s=1)
        axes[0].set_title("Training Data")
        axes[0].set_xlabel("Latent Dimension 1")
        axes[0].set_ylabel("Latent Dimension 2")
        axes[0].grid(True, alpha=0.3)

        # Validation data
        axes[1].scatter(
            val_latent[:, 0], val_latent[:, 1], alpha=0.6, s=1, color="orange"
        )
        axes[1].set_title("Validation Data")
        axes[1].set_xlabel("Latent Dimension 1")
        axes[1].set_ylabel("Latent Dimension 2")
        axes[1].grid(True, alpha=0.3)

        # Test data
        axes[2].scatter(
            test_latent[:, 0], test_latent[:, 1], alpha=0.6, s=1, color="green"
        )
        axes[2].set_title("Test Data")
        axes[2].set_xlabel("Latent Dimension 1")
        axes[2].set_ylabel("Latent Dimension 2")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(results_dir / "latent_space_2d.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot 3: Latent dimensions distribution
    n_dims_to_plot = LATENT_DIM

    # Calculate grid dimensions to fit all latent dimensions
    if LATENT_DIM <= 6:
        nrows, ncols = 2, 3
    elif LATENT_DIM <= 12:
        nrows, ncols = 3, 4
    elif LATENT_DIM <= 16:
        nrows, ncols = 4, 4
    elif LATENT_DIM <= 20:
        nrows, ncols = 4, 5
    else:
        nrows, ncols = 5, 6  # For up to 30 dimensions

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    fig.suptitle(
        f"Latent Dimensions Distribution (All {LATENT_DIM} Dimensions)", fontsize=16
    )
    axes = axes.flatten()

    for i in range(n_dims_to_plot):
        axes[i].hist(
            train_latent[:, i], bins=50, alpha=0.7, label="Train", density=True
        )
        axes[i].hist(val_latent[:, i], bins=50, alpha=0.7, label="Val", density=True)
        axes[i].set_title(f"Latent Dimension {i+1}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    total_subplots = nrows * ncols
    for i in range(n_dims_to_plot, total_subplots):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(results_dir / "latent_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: Reconstruction quality (sample reconstructions)
    model.eval()
    with torch.no_grad():
        # Sample some test data (limit to 4 samples to match subplot grid)
        n_samples = min(4, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        sample_data = X_test_tensor[sample_indices]
        sample_recon, _, _, _ = model(sample_data)

        sample_data_np = sample_data.cpu().numpy()
        sample_recon_np = sample_recon.cpu().numpy()

        # Calculate reconstruction errors
        recon_errors = np.mean((sample_data_np - sample_recon_np) ** 2, axis=1)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("Original vs Reconstructed (Test Set)", fontsize=16)

        for i in range(n_samples):
            # Original
            axes[0, i].plot(sample_data_np[i], alpha=0.8, label="Original")
            axes[0, i].plot(
                sample_recon_np[i], alpha=0.8, label="Reconstructed", linestyle="--"
            )
            axes[0, i].set_title(f"Sample {i+1}\nRecon Error: {recon_errors[i]:.4f}")
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

            # Error
            error = sample_data_np[i] - sample_recon_np[i]
            axes[1, i].plot(error, alpha=0.8, color="red")
            axes[1, i].set_title(f"Reconstruction Error")
            axes[1, i].set_ylabel("Error")
            axes[1, i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_samples, 4):
            axes[0, i].set_visible(False)
            axes[1, i].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        results_dir / "reconstruction_examples.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 5: Time series of latent factors (if dates available)
    if "train_dates" in locals() and LATENT_DIM >= 1:
        # Show up to 6 latent factors for readability
        n_factors_to_plot = min(LATENT_DIM, 6)

        fig, axes = plt.subplots(
            n_factors_to_plot, 1, figsize=(15, 3 * n_factors_to_plot)
        )
        fig.suptitle(
            f"Latent Factors Over Time (First {n_factors_to_plot} of {LATENT_DIM})",
            fontsize=16,
        )

        # Handle single subplot case
        if n_factors_to_plot == 1:
            axes = [axes]

        # Combine all data for plotting
        all_latent = np.vstack([train_latent, val_latent, test_latent])
        all_dates = np.concatenate([train_dates, val_dates, test_dates])

        for i in range(n_factors_to_plot):
            axes[i].plot(all_dates, all_latent[:, i], alpha=0.8)
            axes[i].set_title(f"Latent Factor {i+1}")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, alpha=0.3)

            # Add vertical lines to separate train/val/test
            axes[i].axvline(
                x=train_dates[-1],
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Train/Val Split",
            )
            axes[i].axvline(
                x=val_dates[-1],
                color="orange",
                linestyle="--",
                alpha=0.7,
                label="Val/Test Split",
            )
            if i == 0:
                axes[i].legend()

        axes[-1].set_xlabel("Date")
        plt.tight_layout()
        plt.savefig(results_dir / "latent_timeseries.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("✅ All plots saved")

    # 7. Create summary report
    summary_report = f"""
# VAE Training Summary Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results folder: {results_dir}

## Model Configuration
- Input Dimension: {INPUT_DIM}
- Hidden Dimension: {HIDDEN_DIM}
- Latent Dimension: {LATENT_DIM}
- Dropout Rate: {DROPOUT_RATE}
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}

## Training Configuration
- Learning Rate: {LEARNING_RATE}
- Batch Size: {BATCH_SIZE}
- Max Epochs: {NUM_EPOCHS}
- Patience: {PATIENCE}
- Weight Decay: {WEIGHT_DECAY}

## Loss Configuration
- Beta (KL weight): {BETA}
- C (KL capacity): {C}
- Lambda (orthogonality): {LAMBDA_ORTHO}

## Training Results
- Best Validation Loss: {best_val_loss:.6f}
- Epochs Completed: {len(history["train_loss"])}
- Training Time: {total_time/60:.1f} minutes
- Final Train Loss: {history["train_loss"][-1]:.6f}
- Final Val Loss: {history["val_loss"][-1]:.6f}

## Data Information
- Training Samples: {len(X_train)}
- Validation Samples: {len(X_val)}
- Test Samples: {len(X_test)}
- Features: {INPUT_DIM}
- Compression Ratio: {INPUT_DIM/LATENT_DIM:.1f}:1

## Saved Files
- model_weights.pth: Model state dict
- model_complete.pth: Complete model object
- training_config.json: Configuration and metadata
- training_history.csv/pkl: Training metrics per epoch
- train_latent.npy, val_latent.npy, test_latent.npy: Latent representations
- train_mean.npy, val_mean.npy, test_mean.npy: Latent means
- train_logvar.npy, val_logvar.npy, test_logvar.npy: Latent log variances
- date_info.json: Date information
- *.png: Various plots and visualizations

## Usage
To load this trained model:
```python
import torch
model = torch.load('{results_dir}/model_complete.pth')
# or
model = SimpleVAE(...)
model.load_state_dict(torch.load('{results_dir}/model_weights.pth'))
```

To load latent representations:
```python
import numpy as np
train_latent = np.load('{results_dir}/train_latent.npy')
```
"""

    with open(results_dir / "SUMMARY.md", "w") as f:
        f.write(summary_report)

    print(f"\n🎉 All outputs saved successfully to: {results_dir}")
    print(f"📋 Summary report saved as: {results_dir}/SUMMARY.md")

else:
    print("❌ Cannot start training - model not created")
