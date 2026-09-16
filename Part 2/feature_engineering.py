import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import sys

logger = logging.getLogger(__name__)


def configure_logging(debug_mode=False, log_file="pipeline.log"):
    """Configure logging with optional debug mode."""
    log_level = logging.DEBUG if debug_mode else logging.INFO

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler - always captures all messages
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler - respects debug mode
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def load_cleaned_data(filepath):
    """
    Load the cleaned dataset from pipeline.

    Args:
        filepath: Path to cleaned CSV file

    Returns:
        DataFrame with cleaned data
    """
    try:
        df = pd.read_csv(filepath)
        df['date_time'] = pd.to_datetime(df['date_time'])
        logger.info(f"Successfully loaded cleaned data: {filepath}")
        return df
    except Exception:
        logger.error(f"Failed to load cleaned data from {filepath}", exc_info=True)
        sys.exit(1)


def create_time_features(df):
    """
    Create time-based features from date_time column.
    Includes: hour, day_of_week, is_weekend, and cyclical encoding (sine/cosine).

    Args:
        df: Input DataFrame with date_time column

    Returns:
        DataFrame with added time features
    """
    logger.info("Creating time features...")

    # Extract hour (0-23)
    df['hour'] = df['date_time'].dt.hour
    logger.debug(f"Hour range: {df['hour'].min()} to {df['hour'].max()}")

    # Extract day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['date_time'].dt.dayofweek
    logger.debug(f"Day of week range: {df['day_of_week'].min()} to {df['day_of_week'].max()}")

    # Weekend indicator (Saturday=5, Sunday=6)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    weekend_count = df['is_weekend'].sum()
    logger.debug(f"Weekend records: {weekend_count} out of {len(df)}")

    # Cyclical encoding of hour using sine/cosine (captures 24-hour periodicity)
    # This converts circular hour values to linear coordinates
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    logger.debug(f"Hour cyclical encoding: sin range [{df['hour_sin'].min():.4f}, {df['hour_sin'].max():.4f}], "
                f"cos range [{df['hour_cos'].min():.4f}, {df['hour_cos'].max():.4f}]")

    # Cyclical encoding of day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    logger.debug(f"Day of week cyclical encoding applied")

    logger.info("Time features created: hour, day_of_week, is_weekend, hour_sin, hour_cos, dow_sin, dow_cos")

    return df


def create_weather_features(df):
    """
    Create and encode weather-based features.
    Encodes categorical weather variables and creates derived indicators.

    Args:
        df: Input DataFrame with weather columns

    Returns:
        DataFrame with added weather features
    """
    logger.info("Creating weather features...")

    # One-hot encode weather_main categorical variable
    weather_dummies = pd.get_dummies(df['weather_main'], prefix='weather', drop_first=False)
    logger.debug(f"Weather categories encoded: {list(weather_dummies.columns)}")
    logger.debug(f"Weather category distribution:\n{df['weather_main'].value_counts().to_dict()}")

    df = pd.concat([df, weather_dummies], axis=1)

    # Create derived weather indicator: is_clear (1 if weather is Clear, 0 otherwise)
    df['is_clear'] = (df['weather_main'] == 'Clear').astype(int)
    clear_count = df['is_clear'].sum()
    logger.debug(f"Clear weather records: {clear_count} out of {len(df)}")

    # Create derived weather indicator: is_rainy (1 if weather contains Rain, 0 otherwise)
    df['is_rainy'] = df['weather_main'].str.contains('Rain', case=False, na=False).astype(int)
    rainy_count = df['is_rainy'].sum()
    logger.debug(f"Rainy weather records: {rainy_count} out of {len(df)}")

    # Create cloud coverage intensity indicator (high clouds if > 75%)
    df['high_cloud_coverage'] = (df['clouds_all'] > 75).astype(int)
    high_cloud_count = df['high_cloud_coverage'].sum()
    logger.debug(f"High cloud coverage records (>75%): {high_cloud_count} out of {len(df)}")

    # Create precipitation indicator (rain_1h > 0 or snow_1h > 0)
    df['has_precipitation'] = ((df['rain_1h'] > 0) | (df['snow_1h'] > 0)).astype(int)
    precip_count = df['has_precipitation'].sum()
    logger.debug(f"Records with precipitation: {precip_count} out of {len(df)}")

    logger.info("Weather features created: weather one-hot encoding, is_clear, is_rainy, "
               "high_cloud_coverage, has_precipitation")

    return df


def create_normalized_features(df):
    """
    Create normalized/scaled versions of continuous variables.
    Scales at least two continuous variables using StandardScaler.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with added normalized features
    """
    logger.info("Creating normalized features...")

    scaler = StandardScaler()

    # Normalize temperature (Kelvin to standardized scale)
    temp_original_stats = {
        'mean': df['temp'].mean(),
        'std': df['temp'].std(),
        'min': df['temp'].min(),
        'max': df['temp'].max()
    }
    logger.debug(f"Temperature before scaling - Mean: {temp_original_stats['mean']:.2f}, "
                f"Std: {temp_original_stats['std']:.2f}, "
                f"Min: {temp_original_stats['min']:.2f}, Max: {temp_original_stats['max']:.2f}")

    df['temp_normalized'] = scaler.fit_transform(df[['temp']])
    logger.debug(f"Temperature normalized - Mean: {df['temp_normalized'].mean():.6f}, "
                f"Std: {df['temp_normalized'].std():.6f}")

    # Normalize traffic volume
    traffic_original_stats = {
        'mean': df['traffic_volume'].mean(),
        'std': df['traffic_volume'].std(),
        'min': df['traffic_volume'].min(),
        'max': df['traffic_volume'].max()
    }
    logger.debug(f"Traffic volume before scaling - Mean: {traffic_original_stats['mean']:.0f}, "
                f"Std: {traffic_original_stats['std']:.0f}, "
                f"Min: {traffic_original_stats['min']:.0f}, Max: {traffic_original_stats['max']:.0f}")

    df['traffic_volume_normalized'] = scaler.fit_transform(df[['traffic_volume']])
    logger.debug(f"Traffic volume normalized - Mean: {df['traffic_volume_normalized'].mean():.6f}, "
                f"Std: {df['traffic_volume_normalized'].std():.6f}")

    # Normalize cloud coverage (already 0-100 scale, but standardize anyway)
    cloud_original_stats = {
        'mean': df['clouds_all'].mean(),
        'std': df['clouds_all'].std(),
        'min': df['clouds_all'].min(),
        'max': df['clouds_all'].max()
    }
    logger.debug(f"Cloud coverage before scaling - Mean: {cloud_original_stats['mean']:.2f}, "
                f"Std: {cloud_original_stats['std']:.2f}, "
                f"Min: {cloud_original_stats['min']:.2f}, Max: {cloud_original_stats['max']:.2f}")

    df['clouds_all_normalized'] = scaler.fit_transform(df[['clouds_all']])
    logger.debug(f"Cloud coverage normalized - Mean: {df['clouds_all_normalized'].mean():.6f}, "
                f"Std: {df['clouds_all_normalized'].std():.6f}")

    logger.info("Normalized features created: temp_normalized, traffic_volume_normalized, clouds_all_normalized")

    return df


def create_traffic_target(df):
    """
    Create a data-driven congestion category based on traffic volume distribution.
    Uses quartile-based thresholds to categorize traffic into 4 levels.

    Logic:
    - Calculate Q1 (25%), Q2 (50% - median), Q3 (75%) of traffic_volume
    - Category 0: Volume <= Q1 (Light traffic)
    - Category 1: Q1 < Volume <= Q2 (Moderate traffic)
    - Category 2: Q2 < Volume <= Q3 (Heavy traffic)
    - Category 3: Volume > Q3 (Very heavy traffic)

    Args:
        df: Input DataFrame with traffic_volume column

    Returns:
        DataFrame with added congestion_category column
    """
    logger.info("Creating traffic congestion target variable...")

    # Calculate quartiles
    q1 = df['traffic_volume'].quantile(0.25)
    q2 = df['traffic_volume'].quantile(0.50)  # median
    q3 = df['traffic_volume'].quantile(0.75)

    logger.debug(f"Traffic volume quartiles - Q1: {q1:.0f}, Q2 (Median): {q2:.0f}, Q3: {q3:.0f}")

    # Create congestion category based on quartiles
    df['congestion_category'] = pd.cut(
        df['traffic_volume'],
        bins=[df['traffic_volume'].min() - 1, q1, q2, q3, df['traffic_volume'].max() + 1],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    # Count records in each category
    category_counts = df['congestion_category'].value_counts().sort_index()
    logger.debug(f"Congestion category distribution:\n{category_counts.to_dict()}")

    # Log detailed category information
    category_info = {
        0: f"Light traffic (volume <= {q1:.0f}): {(df['congestion_category'] == 0).sum()} records",
        1: f"Moderate traffic ({q1:.0f} < volume <= {q2:.0f}): {(df['congestion_category'] == 1).sum()} records",
        2: f"Heavy traffic ({q2:.0f} < volume <= {q3:.0f}): {(df['congestion_category'] == 2).sum()} records",
        3: f"Very heavy traffic (volume > {q3:.0f}): {(df['congestion_category'] == 3).sum()} records"
    }
    for cat, info in category_info.items():
        logger.debug(info)

    logger.info("Congestion target created: congestion_category (0=Light, 1=Moderate, 2=Heavy, 3=Very Heavy)")
    logger.info(f"Target distribution: {dict(category_counts)}")

    return df


def engineer_features(input_filepath, output_filepath=None, debug_mode=False):
    """
    Execute complete feature engineering pipeline.

    Args:
        input_filepath: Path to cleaned CSV from pipeline
        output_filepath: Path to save feature-engineered dataset
        debug_mode: If True, enables DEBUG-level logging to console

    Returns:
        Feature-engineered DataFrame
    """
    configure_logging(debug_mode=debug_mode)

    logger.info("=" * 70)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 70)

    # Load cleaned data
    df = load_cleaned_data(input_filepath)

    # Log initial shape
    initial_shape = df.shape
    logger.info(f"Initial dataset shape: {initial_shape[0]} rows × {initial_shape[1]} columns")

    # Create all features
    logger.info("\n--- Creating time features ---")
    df = create_time_features(df)

    logger.info("\n--- Creating weather features ---")
    df = create_weather_features(df)

    logger.info("\n--- Creating normalized features ---")
    df = create_normalized_features(df)

    logger.info("\n--- Creating traffic target variable ---")
    df = create_traffic_target(df)

    # Log final shape
    final_shape = df.shape
    logger.info(f"\nFinal dataset shape: {final_shape[0]} rows × {final_shape[1]} columns")
    logger.info(f"Features added: {final_shape[1] - initial_shape[1]} new columns")

    # Save feature-engineered dataset
    if output_filepath:
        try:
            df.to_csv(output_filepath, index=False)
            logger.info(f"\nFeature-engineered data saved to {output_filepath}")
        except Exception:
            logger.error(f"Failed to save feature-engineered data to {output_filepath}", exc_info=True)
            sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("Feature engineering pipeline completed successfully")
    logger.info("=" * 70)

    return df


if __name__ == "__main__":
    # Run feature engineering pipeline
    input_path = "Metro_Interstate_Traffic_Volume_cleaned.csv"
    output_path = "Metro_Interstate_Traffic_Volume_features.csv"

    # Set debug_mode=True to see DEBUG-level messages in console
    features_df = engineer_features(input_path, output_path, debug_mode=False)
