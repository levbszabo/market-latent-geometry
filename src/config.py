# 🏗️ MODEL ARCHITECTURE HYPERPARAMETERS
# ===============================================================================
# These control the VAE model structure and capacity

# INPUT_DIM - Automatically calculated from data (num_stocks × features_per_stock)
# This will be set automatically when data is loaded

# HIDDEN_DIM - Size of hidden layers in encoder/decoder networks
#   📈 IMPACT: Controls model capacity and learning ability
#   🎯 USAGE:
#   • 64-128:   Small model, fast training, less capacity, good for simple patterns
#   • 256-512:  Medium model, balanced performance (recommended for most cases)
#   • 1024+:    Large model, high capacity, slower training, captures complex patterns
#   💡 TIP: Start with 128-256, increase if underfitting, decrease if overfitting
HIDDEN_DIM = 128
# LATENT_DIM - Dimensionality of latent factor space (the "market factors")
#   📈 IMPACT: Number of independent market factors the model will discover
#   🎯 USAGE:
#   • 4-8:      Very compact, highly interpretable factors (like PCA components)
#   • 8-12:     Good balance for factor models (recommended for finance)
#   • 16-32:    More factors, captures finer market nuances
#   • 32+:      High dimensional, may capture noise
#   💡 TIP: Financial literature suggests 5-15 meaningful market factors exist
LATENT_DIM = 12

# DROPOUT_RATE - Regularization strength during training (prevents overfitting)
#   📈 IMPACT: Higher values = more regularization, less overfitting, may underfit
#   🎯 USAGE:
#   • 0.0:      No dropout, maximum capacity, risk of overfitting
#   • 0.1-0.2:  Light regularization (recommended for most cases)
#   • 0.3-0.5:  Heavy regularization, use if severe overfitting
#   • 0.5+:     Very aggressive, likely to underfit
#   💡 TIP: Start with 0.1, increase if validation loss >> training loss
DROPOUT_RATE = 0.1


# 🚀 TRAINING HYPERPARAMETERS
# ===============================================================================
# These control how the model learns from data

# BATCH_SIZE - Number of samples processed together in each training step
#   📈 IMPACT: Affects training stability, memory usage, and generalization
#   🎯 USAGE:
#   • 16-32:    Small batches, more noise, better generalization, slower training
#   • 64-128:   Medium batches, good balance (recommended)
#   • 256+:     Large batches, stable gradients, may overfit, faster training
#   💡 TIP: Financial data benefits from smaller batches (16-64)
BATCH_SIZE = 32

# LEARNING_RATE - How fast the model learns (step size for gradient descent)
#   📈 IMPACT: Too high = unstable training, too low = slow convergence
#   🎯 USAGE:
#   • 1e-5:     Very slow, stable, good for fine-tuning or sensitive data
#   • 1e-4:     Slow but safe (recommended for financial data)
#   • 1e-3:     Standard rate, good for most deep learning
#   • 1e-2+:    Fast learning, likely to diverge with financial data
#   💡 TIP: Financial data is noisy - use conservative learning rates
LEARNING_RATE = 1e-4

# NUM_EPOCHS - Maximum number of complete passes through the training data
#   📈 IMPACT: More epochs = more training, but risk of overfitting
#   🎯 USAGE:
#   • 50-100:   Quick training, may underfit, good for experimentation
#   • 200-500:  Thorough training (recommended for production)
#   • 1000+:    Very thorough, high risk of overfitting without early stopping
#   💡 TIP: Use early stopping, so this is just an upper bound
NUM_EPOCHS = 80

# PATIENCE - Early stopping patience (epochs without validation improvement)
#   📈 IMPACT: Higher patience = more training before stopping
#   🎯 USAGE:
#   • 10-20:    Aggressive early stopping, prevents overfitting
#   • 50-100:   Moderate patience (recommended)
#   • 200+:     Very patient, allows thorough training
#   💡 TIP: Set to 10-20% of NUM_EPOCHS
PATIENCE = 200

# WEIGHT_DECAY - L2 regularization strength (penalizes large weights)
#   📈 IMPACT: Prevents overfitting by keeping weights small
#   🎯 USAGE:
#   • 0.0:      No weight decay, maximum model flexibility
#   • 1e-5:     Very light regularization
#   • 1e-4:     Light regularization (recommended)
#   • 1e-3:     Strong regularization, use if overfitting
#   • 1e-2+:    Very strong, likely to underfit
WEIGHT_DECAY = 1e-4

# 📊 VAE LOSS FUNCTION HYPERPARAMETERS
# ===============================================================================
# These control the VAE objective function and what the model optimizes for

# BETA - KL divergence weight (β-VAE parameter)
#   📈 IMPACT: Controls trade-off between reconstruction and regularization
#   🎯 USAGE:
#   • 0.0:      Pure autoencoder, no Gaussian prior
#   • 0.1-1.0:  Light VAE regularization, some structure
#   • 1.0:      Standard VAE, forces Gaussian latent space
#   • 4.0+:     β-VAE for disentangled representations
#   💡 TIP: Financial regimes aren't Gaussian - use BETA=0.0 for best results
BETA = 0.00

# C - KL capacity target for gradual annealing
#   📈 IMPACT: Controls how much Gaussian structure to enforce
#   🎯 USAGE:
#   • 0.0:      No capacity constraint (recommended with BETA=0.0)
#   • 0.1-0.5:  Light capacity constraint
#   • 1.0+:     Strong capacity constraint
#   💡 TIP: Keep at 0.0 when BETA=0.0
C = 0.00
# LAMBDA_ORTHO - Orthogonality penalty strength (decorrelates latent factors)
#   📈 IMPACT: Encourages independent, uncorrelated latent factors
#   🎯 USAGE:
#   • 0.0:      No orthogonality constraint
#   • 1e-4:     Very light decorrelation
#   • 1e-3:     Light decorrelation (recommended)
#   • 1e-2:     Strong decorrelation, may hurt reconstruction
#   • 1e-1+:    Very strong, likely to hurt performance
#   💡 TIP: Helps create interpretable, independent market factors
LAMBDA_ORTHO = 1e-3
