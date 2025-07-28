#!/usr/bin/env python3
"""
Simple Dataset Builder for VAE Analysis

We include this script so you can build your own dataset on your own universe/time series data.
**note this script can take a while specifically to write the numpy arrays to disk.**

This script builds a simplified dataset using price data and sector information for S&P 500 stocks.
It focuses on creating a clean, VAE-ready dataset with the following features:
- Log price changes (daily returns)
- Log volume
- Sector encoding (scaled to [0,1] range)

The output format is N x 3 (where N = number of stocks), suitable for VAE dimensionality reduction.
For 500 stocks, this creates a 1500-length feature vector per time step.

Key features:
1. Loads S&P 500 tickers and their price/sector data
2. Calculates log price changes, log volume, and encodes sectors numerically
3. Creates 80:10:10 train/validation/test splits (time-based)
4. Applies global Z-normalization to numeric features based on training data
5. Outputs data in a format suitable for VAE processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class SimpleDatasetBuilder:
    """
    Builds a simplified dataset using price data and sector information for VAE analysis.
    """

    def __init__(
        self, data_path: str = "../data", output_path: str = "../processed_data_simple"
    ):
        self.data_path = Path(data_path)

        # Create timestamped output directory for experiment tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_path = Path(output_path)
        self.output_path = self.base_output_path / timestamp
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Create/update 'latest' symlink for convenience
        latest_path = self.base_output_path / "latest"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(timestamp)

        logger.info(f"Output directory: {self.output_path}")
        logger.info(f"Latest symlink: {latest_path}")

        self.tickers, self.sector_dict = self._load_sp500_tickers_and_sectors()
        self.trading_calendar = None  # Will be created from actual data

    def _load_sp500_tickers_and_sectors(self) -> Tuple[List[str], Dict[str, float]]:
        """Loads the list of S&P 500 tickers and creates sector encoding."""
        logger.info("Loading S&P 500 tickers and sector information...")
        ticker_file = self.data_path / "universe/sp500_tickers.csv"
        if not ticker_file.exists():
            raise FileNotFoundError(f"Ticker file not found: {ticker_file}")

        df = pd.read_csv(ticker_file)

        # Get unique sectors and create mapping
        unique_sectors = sorted(df["sector"].unique())
        logger.info(f"Found {len(unique_sectors)} unique sectors: {unique_sectors}")

        # Create sector to numeric ID mapping (0-based indexing)
        sector_to_id = {sector: idx for idx, sector in enumerate(unique_sectors)}

        # Create ticker to sector mapping with scaled values [0, 1]
        sector_dict = {}
        max_sector_id = len(unique_sectors) - 1

        for _, row in df.iterrows():
            ticker = row["symbol"]
            sector = row["sector"]
            sector_id = sector_to_id[sector]
            # Scale to [0, 1] range
            sector_id_scaled = sector_id / max_sector_id if max_sector_id > 0 else 0.0
            sector_dict[ticker] = sector_id_scaled

        tickers = df["symbol"].unique().tolist()
        logger.info(f"Loaded {len(tickers)} unique tickers with sector mappings.")
        logger.info(
            f"Sector encoding: {dict(list(sector_to_id.items())[:5])}... (showing first 5)"
        )

        return tickers, sector_dict

    def _load_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads and processes price data for a single ticker."""
        price_file = self.data_path / "prices" / f"{ticker}.csv"
        if not price_file.exists():
            logger.warning(f"Price data not found for ticker: {ticker}")
            return None

        df = pd.read_csv(price_file)
        df["date"] = pd.to_datetime(df["date"])

        # Ensure we have the required columns
        required_cols = ["close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column '{col}' for ticker: {ticker}")
                return None
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with missing data
        df = df.dropna(subset=["date", "close", "volume"])

        # Filter out zero or negative prices/volumes
        df = df[(df["close"] > 0) & (df["volume"] > 0)]

        if df.empty:
            logger.warning(f"No valid data remaining for ticker: {ticker}")
            return None

        df = df.sort_values("date").reset_index(drop=True)

        # Calculate log price changes (daily returns)
        df["log_price_change"] = np.log(df["close"] / df["close"].shift(1))

        # Calculate log volume
        df["log_volume"] = np.log(df["volume"])

        # Add sector information
        df["sector_id_scaled"] = self.sector_dict.get(
            ticker, 0.0
        )  # Default to 0.0 if missing

        # Remove the first row (NaN due to price change calculation)
        df = df.dropna(subset=["log_price_change"])

        # Add ticker column
        df["ticker"] = ticker

        return df[
            ["date", "ticker", "log_price_change", "log_volume", "sector_id_scaled"]
        ]

    def _create_master_panel(self) -> pd.DataFrame:
        """Creates the master panel dataset with all tickers."""
        logger.info("Creating master panel dataset...")

        all_data = []
        successful_tickers = []
        failed_tickers = []

        for i, ticker in enumerate(self.tickers):
            logger.info(f"Processing ticker {i+1}/{len(self.tickers)}: {ticker}")
            ticker_data = self._load_price_data(ticker)

            if ticker_data is not None and not ticker_data.empty:
                all_data.append(ticker_data)
                successful_tickers.append(ticker)
                logger.info(
                    f"✅ Successfully processed {ticker}: {len(ticker_data)} rows"
                )
            else:
                failed_tickers.append(ticker)
                logger.warning(f"❌ Failed to process {ticker}")

        logger.info(
            f"Processing complete: {len(successful_tickers)} successful, {len(failed_tickers)} failed"
        )

        if not all_data:
            raise ValueError("No valid data found for any ticker")

        # Combine all ticker data
        panel = pd.concat(all_data, ignore_index=True)
        logger.info(
            f"Created panel with {len(panel)} rows for {panel['ticker'].nunique()} tickers"
        )

        # Update tickers list to only include successful ones
        self.tickers = successful_tickers

        return panel

    def _create_time_splits(
        self, panel: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Creates time-based train/validation/test splits (80:10:10)."""
        logger.info("Creating time-based splits...")

        # Get unique dates and sort them
        unique_dates = sorted(panel["date"].unique())
        n_dates = len(unique_dates)

        # Calculate split indices
        train_end_idx = int(n_dates * 0.8)
        val_end_idx = int(n_dates * 0.9)

        train_end_date = unique_dates[train_end_idx - 1]
        val_end_date = unique_dates[val_end_idx - 1]

        # Create splits
        train_df = panel[panel["date"] <= train_end_date].copy()
        val_df = panel[
            (panel["date"] > train_end_date) & (panel["date"] <= val_end_date)
        ].copy()
        test_df = panel[panel["date"] > val_end_date].copy()

        logger.info(
            f"Train split: {len(train_df)} rows, dates {train_df['date'].min()} to {train_df['date'].max()}"
        )
        logger.info(
            f"Validation split: {len(val_df)} rows, dates {val_df['date'].min()} to {val_df['date'].max()}"
        )
        logger.info(
            f"Test split: {len(test_df)} rows, dates {test_df['date'].min()} to {test_df['date'].max()}"
        )

        return train_df, val_df, test_df

    def _normalize_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Applies global Z-normalization to numeric features using training data statistics."""
        logger.info("Applying global Z-normalization based on training data...")

        # Only normalize numeric features - sector_id_scaled is already in [0,1] range
        feature_cols = ["log_price_change", "log_volume"]

        # Calculate global normalization parameters from training data only
        logger.info("Computing global normalization statistics...")
        global_mean = train_df[feature_cols].mean()
        global_std = (
            train_df[feature_cols].std().clip(lower=1e-8)
        )  # Prevent division by zero

        logger.info("Global normalization statistics:")
        for col in feature_cols:
            logger.info(
                f"  {col}: mean={global_mean[col]:.6f}, std={global_std[col]:.6f}"
            )

        # Apply global normalization to all splits
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            logger.info(f"Normalizing {split_name} split...")
            df[feature_cols] = (df[feature_cols] - global_mean) / global_std

        # Save normalization statistics for future use
        logger.info("Saving global normalization statistics...")
        normalization_stats = pd.DataFrame(
            {"feature": feature_cols, "mean": global_mean, "std": global_std}
        )
        normalization_stats.to_csv(
            self.output_path / "normalization_stats.csv", index=False
        )

        logger.info("Global normalization completed")
        logger.info(f"Sector features remain in original [0,1] range")

        return train_df, val_df, test_df

    def _convert_to_vae_format(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Converts panel data to VAE-ready format: N x 3 (where N = number of stocks).
        Returns array of shape (time_steps, n_stocks * 3) and metadata.
        """
        logger.info("Converting data to VAE format...")

        # Get unique dates and tickers
        unique_dates = sorted(df["date"].unique())
        unique_tickers = sorted(self.tickers)

        n_dates = len(unique_dates)
        n_tickers = len(unique_tickers)
        n_features = 3  # log_price_change, log_volume, sector_id_scaled

        logger.info(f"Creating array of shape ({n_dates}, {n_tickers * n_features})")
        logger.info(
            f"This represents {n_dates} time steps with {n_tickers} stocks × {n_features} features each"
        )

        # Initialize array
        vae_array = np.full((n_dates, n_tickers * n_features), np.nan)

        # Create feature names for reference
        feature_names = []
        for ticker in unique_tickers:
            feature_names.extend(
                [
                    f"{ticker}_log_price_change",
                    f"{ticker}_log_volume",
                    f"{ticker}_sector_id_scaled",
                ]
            )

        # Fill the array
        for date_idx, date in enumerate(unique_dates):
            date_data = df[df["date"] == date]

            for ticker_idx, ticker in enumerate(unique_tickers):
                ticker_data = date_data[date_data["ticker"] == ticker]

                if not ticker_data.empty:
                    # Place features for this ticker
                    base_idx = ticker_idx * n_features
                    row = ticker_data.iloc[0]
                    vae_array[date_idx, base_idx] = row["log_price_change"]
                    vae_array[date_idx, base_idx + 1] = row["log_volume"]
                    vae_array[date_idx, base_idx + 2] = row["sector_id_scaled"]

        # Handle missing values by forward filling then backward filling
        vae_df = pd.DataFrame(vae_array, columns=feature_names, index=unique_dates)
        vae_df = vae_df.fillna(method="ffill").fillna(method="bfill")

        # If still any NaN values, fill with 0
        vae_array = vae_df.fillna(0).values

        logger.info(f"Final VAE array shape: {vae_array.shape}")
        logger.info(f"Number of NaN values: {np.isnan(vae_array).sum()}")

        return vae_array, feature_names, unique_dates

    def run_pipeline(self):
        """Runs the complete simple dataset creation pipeline."""
        logger.info("Starting simple dataset creation pipeline...")

        # Create master panel
        panel = self._create_master_panel()

        # Create time-based splits
        train_df, val_df, test_df = self._create_time_splits(panel)

        # Normalize features
        train_df, val_df, test_df = self._normalize_features(train_df, val_df, test_df)

        # Save split datasets
        logger.info("Saving split datasets...")
        train_df.to_parquet(self.output_path / "train_simple.parquet")
        val_df.to_parquet(self.output_path / "validation_simple.parquet")
        test_df.to_parquet(self.output_path / "test_simple.parquet")

        # Convert to VAE format
        logger.info("Converting datasets to VAE format...")

        train_vae, feature_names, train_dates = self._convert_to_vae_format(train_df)
        val_vae, _, val_dates = self._convert_to_vae_format(val_df)
        test_vae, _, test_dates = self._convert_to_vae_format(test_df)

        # Save VAE-ready arrays
        logger.info("Saving VAE-ready datasets...")
        np.save(self.output_path / "train_vae.npy", train_vae)
        np.save(self.output_path / "validation_vae.npy", val_vae)
        np.save(self.output_path / "test_vae.npy", test_vae)

        # Save metadata
        metadata = {
            "feature_names": feature_names,
            "n_tickers": len(self.tickers),
            "n_features_per_ticker": 3,
            "total_features": len(feature_names),
            "tickers": self.tickers,
            "train_dates": [str(d) for d in train_dates],
            "val_dates": [str(d) for d in val_dates],
            "test_dates": [str(d) for d in test_dates],
            "normalization": "global_zscore",
            "sector_encoding": "scaled_id_0_to_1",
            "features": ["log_price_change", "log_volume", "sector_id_scaled"],
        }

        import json

        with open(self.output_path / "dataset_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Validation checks
        logger.info("\n" + "=" * 50)
        logger.info("DATASET VALIDATION CHECKS")
        logger.info("=" * 50)

        # Shape validation
        expected_features = len(self.tickers) * 3
        assert (
            train_vae.shape[1] == expected_features
        ), f"Expected {expected_features} features, got {train_vae.shape[1]}"
        logger.info(
            f"✅ Shape validation passed: {train_vae.shape[1]} == {expected_features}"
        )

        # Feature scaling validation
        # Check numeric features (every 3rd starting from 0 and 1)
        price_features = train_vae[:, 0::3]  # log_price_change features
        volume_features = train_vae[:, 1::3]  # log_volume features
        sector_features = train_vae[:, 2::3]  # sector_id_scaled features

        logger.info(
            f"Price features - mean: {price_features.mean():.6f}, std: {price_features.std():.6f}"
        )
        logger.info(
            f"Volume features - mean: {volume_features.mean():.6f}, std: {volume_features.std():.6f}"
        )
        logger.info(
            f"Sector features - min: {sector_features.min():.6f}, max: {sector_features.max():.6f}"
        )

        # Verify normalization
        assert (
            abs(price_features.mean()) < 0.1
        ), f"Price features not properly normalized: mean={price_features.mean()}"
        assert (
            abs(volume_features.mean()) < 0.1
        ), f"Volume features not properly normalized: mean={volume_features.mean()}"
        assert (
            sector_features.min() >= 0.0
        ), f"Sector features below 0: min={sector_features.min()}"
        assert (
            sector_features.max() <= 1.0
        ), f"Sector features above 1: max={sector_features.max()}"

        logger.info("✅ All validation checks passed!")

        # Create summary
        logger.info("\n" + "=" * 50)
        logger.info("DATASET CREATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Successfully processed {len(self.tickers)} tickers")
        logger.info(
            f"Train dataset: {train_vae.shape[0]} time steps × {train_vae.shape[1]} features"
        )
        logger.info(
            f"Validation dataset: {val_vae.shape[0]} time steps × {val_vae.shape[1]} features"
        )
        logger.info(
            f"Test dataset: {test_vae.shape[0]} time steps × {test_vae.shape[1]} features"
        )
        logger.info(
            f"Feature vector length: {train_vae.shape[1]} ({len(self.tickers)} stocks × 3 features)"
        )
        logger.info(
            f"Features per ticker: log_price_change, log_volume, sector_id_scaled"
        )
        logger.info(
            f"Normalization: Global Z-score for numeric features, [0,1] scaling for sectors"
        )
        logger.info(f"Output directory: {self.output_path}")
        logger.info("=" * 50)


def main():
    """Main execution function."""
    builder = SimpleDatasetBuilder(
        data_path="../data", output_path="../processed_data_simple"
    )
    builder.run_pipeline()


if __name__ == "__main__":
    main()
