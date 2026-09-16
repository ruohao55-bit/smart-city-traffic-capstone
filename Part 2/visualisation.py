import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_logging(log_file="pipeline.log"):
    """Configure logging for visualization pipeline."""
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


def load_feature_engineered_data(filepath):
    """Load the feature-engineered dataset."""
    try:
        df = pd.read_csv(filepath)
        df['date_time'] = pd.to_datetime(df['date_time'])
        logger.info(f"Successfully loaded feature-engineered data: {filepath}")
        logger.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception:
        logger.error(f"Failed to load data from {filepath}", exc_info=True)
        sys.exit(1)


def create_output_dir(output_dir="visualisations"):
    """Create output directory for visualizations."""
    Path(output_dir).mkdir(exist_ok=True)
    logger.info(f"Output directory ready: {output_dir}")
    return output_dir


def visualize_traffic_by_hour(df, output_dir):
    """
    Visualization 1: Traffic Demand by Hour of Day

    Line plot showing average traffic volume for each hour (0-23).
    Reveals daily traffic patterns and peak hours.

    Interpretation:
    - Identifies rush hours and peak traffic periods
    - Shows variations in traffic demand throughout the day
    - Helps understand when congestion is most likely
    """
    logger.info("\n--- Creating Visualization 1: Traffic Demand by Hour ---")

    plt.figure(figsize=(14, 6))

    # Calculate mean traffic volume by hour
    hourly_traffic = df.groupby('hour')['traffic_volume'].agg(['mean', 'std', 'count'])
    logger.debug(f"Hourly statistics calculated. Hours with data: {len(hourly_traffic)}")

    # Plot with error bars (std dev)
    plt.plot(hourly_traffic.index, hourly_traffic['mean'], marker='o', linewidth=2.5,
             markersize=8, color='#2E86AB', label='Mean Traffic Volume')
    plt.fill_between(hourly_traffic.index,
                      hourly_traffic['mean'] - hourly_traffic['std'],
                      hourly_traffic['mean'] + hourly_traffic['std'],
                      alpha=0.2, color='#2E86AB', label='±1 Std Dev')

    plt.xlabel('Hour of Day', fontsize=12, fontweight='bold')
    plt.ylabel('Traffic Volume', fontsize=12, fontweight='bold')
    plt.title('Traffic Demand by Hour of Day', fontsize=14, fontweight='bold')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    output_path = f"{output_dir}/01_traffic_by_hour.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved: {output_path}")

    # Log interpretation
    peak_hour = hourly_traffic['mean'].idxmax()
    peak_volume = hourly_traffic['mean'].max()
    logger.info(f"Interpretation: Peak traffic occurs at hour {peak_hour} with average volume {peak_volume:.0f}")

    plt.close()


def visualize_weekday_vs_weekend(df, output_dir):
    """
    Visualization 2: Weekday vs Weekend Traffic Patterns

    Box plot comparing traffic volume distribution between weekdays and weekends.
    Reveals differences in traffic behavior based on day type.

    Interpretation:
    - Shows if weekends have less congestion than weekdays
    - Displays variability in traffic patterns
    - Helps distinguish commuter patterns from leisure travel
    """
    logger.info("\n--- Creating Visualization 2: Weekday vs Weekend Traffic ---")

    plt.figure(figsize=(10, 6))

    # Create labels for weekday/weekend
    day_labels = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    df['day_label'] = df['day_of_week'].map(day_labels)
    df['day_type'] = df['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})

    # Calculate statistics
    weekday_stats = df[df['is_weekend'] == 0]['traffic_volume'].describe()
    weekend_stats = df[df['is_weekend'] == 1]['traffic_volume'].describe()
    logger.debug(f"Weekday traffic - Mean: {weekday_stats['mean']:.0f}, Median: {weekday_stats['50%']:.0f}")
    logger.debug(f"Weekend traffic - Mean: {weekend_stats['mean']:.0f}, Median: {weekend_stats['50%']:.0f}")

    # Create box plot
    sns.boxplot(data=df, x='day_type', y='traffic_volume', palette=['#2E86AB', '#A23B72'],
                width=0.6, ax=plt.gca())

    plt.xlabel('Day Type', fontsize=12, fontweight='bold')
    plt.ylabel('Traffic Volume', fontsize=12, fontweight='bold')
    plt.title('Traffic Volume: Weekday vs Weekend', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()

    output_path = f"{output_dir}/02_weekday_vs_weekend.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved: {output_path}")

    # Log interpretation
    mean_diff = weekday_stats['mean'] - weekend_stats['mean']
    logger.info(f"Interpretation: Weekday traffic is {mean_diff:.0f} vehicles higher than weekends (average)")

    plt.close()


def visualize_temperature_vs_traffic(df, output_dir):
    """
    Visualization 3: Temperature vs Traffic Volume

    Scatter plot with density showing relationship between temperature and traffic.
    Reveals weather-related impacts on traffic patterns.

    Interpretation:
    - Shows if extreme temperatures affect traffic volume
    - Identifies optimal temperature ranges for traffic flow
    - Helps understand weather sensitivity of traffic patterns
    """
    logger.info("\n--- Creating Visualization 3: Temperature vs Traffic ---")

    plt.figure(figsize=(12, 6))

    # Convert temperature from Kelvin to Celsius for readability
    df['temp_celsius'] = df['temp'] - 273.15

    # Create scatter plot with color gradient by density
    scatter = plt.scatter(df['temp_celsius'], df['traffic_volume'],
                         c=df['clouds_all'], cmap='viridis', alpha=0.5, s=30)

    plt.xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    plt.ylabel('Traffic Volume', fontsize=12, fontweight='bold')
    plt.title('Temperature vs Traffic Volume (colored by Cloud Coverage)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')

    cbar = plt.colorbar(scatter, label='Cloud Coverage (%)')
    cbar.set_label('Cloud Coverage (%)', fontsize=11, fontweight='bold')

    # Add correlation
    correlation = df['temp_celsius'].corr(df['traffic_volume'])
    plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = f"{output_dir}/03_temperature_vs_traffic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved: {output_path}")

    # Log interpretation
    logger.debug(f"Temperature range: {df['temp_celsius'].min():.1f}°C to {df['temp_celsius'].max():.1f}°C")
    logger.info(f"Interpretation: Temperature-traffic correlation is {correlation:.3f} "
               f"({'weak' if abs(correlation) < 0.3 else 'moderate' if abs(correlation) < 0.7 else 'strong'} relationship)")

    plt.close()


def visualize_traffic_distribution(df, output_dir):
    """
    Visualization 4: Traffic Volume Distribution by Congestion Category

    Histogram showing distribution of traffic volumes with categories overlaid.
    Reveals traffic volume distribution and congestion patterns.

    Interpretation:
    - Shows how traffic volumes are distributed
    - Displays balance across congestion categories
    - Identifies whether traffic follows normal distribution
    """
    logger.info("\n--- Creating Visualization 4: Traffic Distribution ---")

    plt.figure(figsize=(12, 6))

    # Define colors for categories
    colors = {0: '#2ECC71', 1: '#F39C12', 2: '#E74C3C', 3: '#C0392B'}
    labels = {0: 'Light', 1: 'Moderate', 2: 'Heavy', 3: 'Very Heavy'}

    # Plot histogram for each category
    for cat in sorted(df['congestion_category'].unique()):
        cat_data = df[df['congestion_category'] == cat]['traffic_volume']
        plt.hist(cat_data, bins=50, alpha=0.6, label=f'{labels[cat]} ({len(cat_data)})',
                color=colors[cat], edgecolor='black', linewidth=0.5)

    plt.xlabel('Traffic Volume', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Traffic Volume Distribution by Congestion Category', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()

    output_path = f"{output_dir}/04_traffic_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved: {output_path}")

    # Log interpretation
    category_counts = df['congestion_category'].value_counts().sort_index()
    logger.debug(f"Category distribution: {dict(category_counts)}")
    logger.info(f"Interpretation: Traffic is relatively balanced across categories "
               f"(25% each), indicating good stratification")

    plt.close()


def visualize_hourly_heatmap(df, output_dir):
    """
    Visualization 5: Hourly Traffic Heatmap by Day of Week

    Heatmap showing average traffic volume for each hour and day combination.
    Reveals complex patterns of traffic across hours and days.

    Interpretation:
    - Shows hourly patterns separately for each day of week
    - Identifies when congestion is worst (darkest regions)
    - Reveals differences between weekdays and weekends
    """
    logger.info("\n--- Creating Visualization 5: Hourly Traffic Heatmap ---")

    plt.figure(figsize=(14, 7))

    # Create pivot table for heatmap
    day_labels = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                  4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    heatmap_data = df.pivot_table(values='traffic_volume',
                                   index='hour',
                                   columns='day_of_week',
                                   aggfunc='mean')
    heatmap_data.columns = [day_labels[i] for i in heatmap_data.columns]

    logger.debug(f"Heatmap created with shape: {heatmap_data.shape}")

    # Create heatmap
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Average Traffic Volume'},
                linewidths=0.5, ax=plt.gca())

    plt.xlabel('Day of Week', fontsize=12, fontweight='bold')
    plt.ylabel('Hour of Day', fontsize=12, fontweight='bold')
    plt.title('Average Traffic Volume Heatmap: Hour × Day of Week', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = f"{output_dir}/05_hourly_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved: {output_path}")

    # Log interpretation
    max_traffic_hour = heatmap_data.mean(axis=1).idxmax()
    max_traffic_day = heatmap_data.mean(axis=0).idxmax()
    logger.debug(f"Peak hour overall: {max_traffic_hour}, Peak day overall: {max_traffic_day}")
    logger.info(f"Interpretation: Heaviest traffic hour is {max_traffic_hour}:00, "
               f"Peak day is {max_traffic_day}")

    plt.close()


def visualize_weather_impact(df, output_dir):
    """
    Visualization 6: Weather Impact on Traffic

    Comparison of traffic volumes across different weather conditions.
    Shows how weather types affect traffic patterns.

    Interpretation:
    - Shows if certain weather conditions increase congestion
    - Reveals weather-related traffic management challenges
    - Identifies safest/most congested weather conditions
    """
    logger.info("\n--- Creating Visualization 6: Weather Impact on Traffic ---")

    plt.figure(figsize=(14, 6))

    # Get top 8 weather conditions
    top_weather = df['weather_main'].value_counts().head(8).index
    weather_data = df[df['weather_main'].isin(top_weather)]

    logger.debug(f"Analyzing {len(top_weather)} most common weather conditions")

    # Create violin plot
    weather_order = weather_data.groupby('weather_main')['traffic_volume'].median().sort_values(ascending=False).index
    sns.violinplot(data=weather_data, x='weather_main', y='traffic_volume', order=weather_order,
                   palette='Set2', ax=plt.gca())

    plt.xlabel('Weather Condition', fontsize=12, fontweight='bold')
    plt.ylabel('Traffic Volume', fontsize=12, fontweight='bold')
    plt.title('Traffic Volume Distribution by Weather Condition', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()

    output_path = f"{output_dir}/06_weather_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved: {output_path}")

    # Log interpretation
    weather_means = weather_data.groupby('weather_main')['traffic_volume'].mean().sort_values(ascending=False)
    logger.debug(f"Weather impacts (mean traffic): {weather_means.to_dict()}")
    logger.info(f"Interpretation: {weather_means.index[0]} shows highest average traffic, "
               f"{weather_means.index[-1]} shows lowest")

    plt.close()


def create_visualisations(input_filepath, output_dir="visualisations"):
    """
    Execute complete visualisation pipeline.

    Args:
        input_filepath: Path to feature-engineered CSV
        output_dir: Directory to save visualizations
    """
    configure_logging()

    logger.info("=" * 70)
    logger.info("Starting visualisation pipeline")
    logger.info("=" * 70)

    # Load data
    df = load_feature_engineered_data(input_filepath)

    # Create output directory
    output_dir = create_output_dir(output_dir)

    # Create visualizations
    logger.info("\n### Creating Visualizations ###")
    visualize_traffic_by_hour(df, output_dir)
    visualize_weekday_vs_weekend(df, output_dir)
    visualize_temperature_vs_traffic(df, output_dir)
    visualize_traffic_distribution(df, output_dir)
    visualize_hourly_heatmap(df, output_dir)
    visualize_weather_impact(df, output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("Visualisation pipeline completed successfully")
    logger.info("=" * 70)
    logger.info(f"All visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    input_path = "Metro_Interstate_Traffic_Volume_features.csv"
    output_directory = "visualisations"

    create_visualisations(input_path, output_directory)
