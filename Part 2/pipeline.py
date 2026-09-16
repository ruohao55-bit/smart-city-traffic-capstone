import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys

logger = logging.getLogger(__name__)


def configure_logging(log_file="pipeline.log"):
    """Configure logging with both file and console handlers."""
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def load_csv(filepath):
    """
    Load the raw CSV file with error handling.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(
            f"Successfully loaded CSV file: {filepath} "
            f"({df.shape[0]} rows, {df.shape[1]} columns)"
        )
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}", exc_info=True)
        sys.exit(1)
    except pd.errors.ParserError:
        logger.error(f"Error parsing CSV file: {filepath}", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error(f"Unexpected error loading CSV file: {filepath}", exc_info=True)
        sys.exit(1)


def validate_schema(df, expected_columns):
    """
    Validate that all expected columns are present in the dataset.

    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names

    Returns:
        True if valid, raises exception otherwise
    """
    missing_columns = set(expected_columns) - set(df.columns)

    if missing_columns:
        logger.error(
            f"Schema validation failed. Missing columns: {missing_columns}"
        )
        sys.exit(1)

    logger.info(
        f"Schema validation passed. All {len(expected_columns)} expected columns present."
    )
    return True


def remove_duplicates(df):
    """
    Identify and remove duplicate rows.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with duplicates removed
    """
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates()
    removed_count = initial_rows - len(df_cleaned)

    if removed_count > 0:
        logger.warning(
            f"Removed {removed_count} duplicate rows. "
            f"Reason: exact row duplicates found. ({initial_rows} rows -> {len(df_cleaned)} rows)"
        )
    else:
        logger.info("No duplicate rows found.")

    return df_cleaned


def parse_datetime(df):
    """
    Parse and validate date/time fields.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with parsed datetime column
    """
    try:
        df['date_time'] = pd.to_datetime(df['date_time'], format='%d/%m/%Y %H:%M')
        logger.info(
            f"Successfully parsed date_time column. "
            f"Date range: {df['date_time'].min()} to {df['date_time'].max()}"
        )
    except Exception:
        logger.error("Failed to parse date_time column", exc_info=True)
        sys.exit(1)

    return df


def standardize_categorical_values(df):
    """
    Standardize inconsistent categorical values.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with standardized categorical values
    """
    changes_made = 0

    # Standardize 'holiday' column - replace 'None' string with actual None
    if 'holiday' in df.columns:
        initial_none_count = (df['holiday'] == 'None').sum()
        df.loc[df['holiday'] == 'None', 'holiday'] = np.nan
        if initial_none_count > 0:
            logger.warning(
                f"Modified {initial_none_count} holiday values. "
                f"Reason: standardized 'None' string values to NaN"
            )
            changes_made += initial_none_count

    # Standardize 'weather_main' - ensure consistent capitalization
    if 'weather_main' in df.columns:
        df['weather_main'] = df['weather_main'].str.strip().str.title()
        logger.info(
            f"Standardized weather_main column: applied title case to {df['weather_main'].nunique()} unique values"
        )

    # Standardize 'weather_description' - consistent capitalization
    if 'weather_description' in df.columns:
        df['weather_description'] = df['weather_description'].str.strip().str.lower()
        logger.info(
            f"Standardized weather_description column: applied lowercase to {df['weather_description'].nunique()} unique values"
        )

    return df


def detect_and_handle_outliers(df):
    """
    Detect and handle outliers or impossible values.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with outliers imputed
    """
    total_imputed = 0

    # Check for temperature at 0 Kelvin or below absolute zero
    if 'temp' in df.columns:
        invalid_temp_mask = df['temp'] < 1  # Below 1K is physically impossible
        invalid_temp_count = invalid_temp_mask.sum()

        if invalid_temp_count > 0:
            logger.warning(
                f"Detected {invalid_temp_count} invalid temperature values "
                f"(less than 1 Kelvin). Reason: physically impossible sensor readings"
            )

            # Impute by month
            df['year_month'] = df['date_time'].dt.to_period('M')
            for month in df['year_month'].unique():
                month_mask = df['year_month'] == month
                valid_temps = df.loc[month_mask & ~invalid_temp_mask, 'temp']

                if len(valid_temps) > 0:
                    median_temp = valid_temps.median()
                    impute_mask = month_mask & invalid_temp_mask
                    df.loc[impute_mask, 'temp'] = median_temp
                    imputed_in_month = impute_mask.sum()
                    if imputed_in_month > 0:
                        logger.warning(
                            f"Imputed {imputed_in_month} temperature outliers in {month}. "
                            f"Reason: monthly median imputation"
                        )
                        total_imputed += imputed_in_month

            df.drop('year_month', axis=1, inplace=True)

    # Check for rainfall exceeding physically plausible range (9000 mm)
    if 'rain_1h' in df.columns:
        invalid_rain_mask = df['rain_1h'] > 9000
        invalid_rain_count = invalid_rain_mask.sum()

        if invalid_rain_count > 0:
            logger.warning(
                f"Detected {invalid_rain_count} invalid rainfall values "
                f"(exceeding 9000 mm). Reason: exceed physically plausible range"
            )

            # Impute by month
            df['year_month'] = df['date_time'].dt.to_period('M')
            for month in df['year_month'].unique():
                month_mask = df['year_month'] == month
                valid_rains = df.loc[month_mask & ~invalid_rain_mask, 'rain_1h']

                if len(valid_rains) > 0:
                    median_rain = valid_rains.median()
                    impute_mask = month_mask & invalid_rain_mask
                    df.loc[impute_mask, 'rain_1h'] = median_rain
                    imputed_in_month = impute_mask.sum()
                    if imputed_in_month > 0:
                        logger.warning(
                            f"Imputed {imputed_in_month} rainfall outliers in {month}. "
                            f"Reason: monthly median imputation"
                        )
                        total_imputed += imputed_in_month

            df.drop('year_month', axis=1, inplace=True)

    # Check for snow values exceeding 9000 mm
    if 'snow_1h' in df.columns:
        invalid_snow_mask = df['snow_1h'] > 9000
        invalid_snow_count = invalid_snow_mask.sum()

        if invalid_snow_count > 0:
            logger.warning(
                f"Detected {invalid_snow_count} invalid snow values "
                f"(exceeding 9000 mm). Reason: exceed physically plausible range"
            )

            # Impute by month
            df['year_month'] = df['date_time'].dt.to_period('M')
            for month in df['year_month'].unique():
                month_mask = df['year_month'] == month
                valid_snows = df.loc[month_mask & ~invalid_snow_mask, 'snow_1h']

                if len(valid_snows) > 0:
                    median_snow = valid_snows.median()
                    impute_mask = month_mask & invalid_snow_mask
                    df.loc[impute_mask, 'snow_1h'] = median_snow
                    imputed_in_month = impute_mask.sum()
                    if imputed_in_month > 0:
                        logger.warning(
                            f"Imputed {imputed_in_month} snow outliers in {month}. "
                            f"Reason: monthly median imputation"
                        )
                        total_imputed += imputed_in_month

            df.drop('year_month', axis=1, inplace=True)

    # Check for negative traffic volume
    if 'traffic_volume' in df.columns:
        negative_traffic_mask = df['traffic_volume'] < 0
        negative_traffic_count = negative_traffic_mask.sum()

        if negative_traffic_count > 0:
            logger.warning(
                f"Detected {negative_traffic_count} negative traffic volume values. "
                f"Reason: impossible negative traffic counts"
            )

            # Impute by month
            df['year_month'] = df['date_time'].dt.to_period('M')
            for month in df['year_month'].unique():
                month_mask = df['year_month'] == month
                valid_volumes = df.loc[month_mask & ~negative_traffic_mask, 'traffic_volume']

                if len(valid_volumes) > 0:
                    median_volume = valid_volumes.median()
                    impute_mask = month_mask & negative_traffic_mask
                    df.loc[impute_mask, 'traffic_volume'] = median_volume
                    imputed_in_month = impute_mask.sum()
                    if imputed_in_month > 0:
                        logger.warning(
                            f"Imputed {imputed_in_month} traffic volume outliers in {month}. "
                            f"Reason: monthly median imputation"
                        )
                        total_imputed += imputed_in_month

            df.drop('year_month', axis=1, inplace=True)

    if total_imputed == 0:
        logger.info("No outliers detected in the dataset.")

    return df


def clean_pipeline(csv_filepath, output_filepath=None):
    """
    Execute the complete data cleaning pipeline.

    Args:
        csv_filepath: Path to input CSV file
        output_filepath: Path to save cleaned CSV (optional)

    Returns:
        Cleaned DataFrame
    """
    logger.info("=" * 70)
    logger.info("Starting data pipeline execution")
    logger.info("=" * 70)

    # Define expected columns
    expected_columns = [
        'holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all',
        'weather_main', 'weather_description', 'date_time', 'traffic_volume'
    ]

    # Step 1: Load CSV
    logger.info("\n--- Step 1: Loading raw CSV file ---")
    df = load_csv(csv_filepath)

    # Step 2: Validate schema
    logger.info("\n--- Step 2: Validating data schema ---")
    validate_schema(df, expected_columns)

    # Step 3: Remove duplicates
    logger.info("\n--- Step 3: Removing duplicate rows ---")
    df = remove_duplicates(df)

    # Step 4: Parse datetime
    logger.info("\n--- Step 4: Parsing date_time column ---")
    df = parse_datetime(df)

    # Step 5: Standardize categorical values
    logger.info("\n--- Step 5: Standardizing categorical values ---")
    df = standardize_categorical_values(df)

    # Step 6: Detect and handle outliers
    logger.info("\n--- Step 6: Detecting and handling outliers ---")
    df = detect_and_handle_outliers(df)

    # Save cleaned data if output path provided
    if output_filepath:
        try:
            df.to_csv(output_filepath, index=False)
            logger.info(
                f"\nCleaned data saved to {output_filepath} "
                f"({df.shape[0]} rows, {df.shape[1]} columns)"
            )
        except Exception:
            logger.error(f"Failed to save cleaned data to {output_filepath}", exc_info=True)
            sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("Data pipeline execution completed successfully")
    logger.info("=" * 70)

    return df


if __name__ == "__main__":
    configure_logging()

    csv_path = "Metro_Interstate_Traffic_Volume.csv"
    output_path = "Metro_Interstate_Traffic_Volume_cleaned.csv"

    cleaned_data = clean_pipeline(csv_path, output_path)
