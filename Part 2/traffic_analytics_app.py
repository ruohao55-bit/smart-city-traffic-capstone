"""
Traffic Analytics CLI Application

A command-line tool for querying and analyzing traffic patterns from the
Metro Interstate Traffic Volume dataset.

Usage:
    python traffic_analytics_app.py <command> [arguments]

Commands:
    query-time       Query traffic for a specific date/time
    peak-hours       Identify high-traffic periods
    weekday-compare  Compare weekday and weekend traffic
    best-times       Find recommended (low-traffic) travel periods
    weather-impact   Analyze weather effects on traffic
    help             Display help information
"""

import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


def configure_logging(log_file="pipeline.log"):
    """Configure logging for the application."""
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

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def load_dataset(filepath: str) -> Optional[pd.DataFrame]:
    """Load the feature-engineered dataset."""
    try:
        df = pd.read_csv(filepath)
        df['date_time'] = pd.to_datetime(df['date_time'])
        logger.info(f"Successfully loaded dataset: {filepath}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return None


def display_help():
    """Display help information."""
    help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    TRAFFIC ANALYTICS APPLICATION                           ║
║                   Query and Analyze Traffic Patterns                       ║
╚════════════════════════════════════════════════════════════════════════════╝

AVAILABLE COMMANDS:

1. query-time <YYYY-MM-DD> <HH:MM>
   ────────────────────────────────────
   Query traffic volume for a specific date and time.

   Example: python traffic_analytics_app.py query-time 2016-01-15 14:30

   Output: Shows traffic volume, weather, temperature, congestion level


2. peak-hours [--limit N]
   ──────────────────────
   Identify the highest traffic periods (hours and days).

   Examples:
   - python traffic_analytics_app.py peak-hours
   - python traffic_analytics_app.py peak-hours --limit 10

   Output: Top N hours with highest average traffic


3. weekday-compare
   ────────────────
   Compare traffic patterns between weekdays and weekends.

   Example: python traffic_analytics_app.py weekday-compare

   Output: Statistics comparison, peak hours for each day type


4. best-times [--max-volume N]
   ─────────────────────────────
   Recommend best times to travel (lowest traffic periods).

   Examples:
   - python traffic_analytics_app.py best-times
   - python traffic_analytics_app.py best-times --max-volume 2000

   Output: Hours and days with traffic below threshold


5. weather-impact
   ───────────────
   Analyze how different weather conditions affect traffic.

   Example: python traffic_analytics_app.py weather-impact

   Output: Average traffic by weather type, rankings


6. help
   ────
   Display this help message.

═══════════════════════════════════════════════════════════════════════════════
"""
    print(help_text)
    logger.info("Help command invoked")


def query_by_time(df: pd.DataFrame, date_str: str, time_str: str) -> None:
    """
    Query traffic for a specific date and time.

    Args:
        df: Feature-engineered dataset
        date_str: Date in YYYY-MM-DD format
        time_str: Time in HH:MM format
    """
    logger.info(f"Command invoked: query-time with date={date_str}, time={time_str}")

    try:
        # Parse date and time
        query_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        logger.debug(f"Parsed datetime: {query_datetime}")
    except ValueError as e:
        error_msg = f"Invalid date/time format. Use YYYY-MM-DD for date and HH:MM for time."
        logger.error(f"User input error: {error_msg}")
        print(f"\n❌ Error: {error_msg}")
        return

    # Extract hour and date for querying
    hour = query_datetime.hour
    query_date = query_datetime.date()

    # Find matching records
    matching_records = df[
        (df['date_time'].dt.hour == hour) &
        (df['date_time'].dt.date == query_date)
    ]

    if matching_records.empty:
        # Try to find closest records
        logger.warning(f"No exact match found for {query_datetime}")
        print(f"\n⚠️  No records found for {date_str} {time_str}")

        # Find records within ±2 hours on the same date
        nearby_records = df[
            (df['date_time'].dt.date == query_date) &
            (df['date_time'].dt.hour.isin(range(max(0, hour - 2), min(24, hour + 3))))
        ]

        if nearby_records.empty:
            print("   No nearby records within ±2 hours on the same date.")
            logger.info(f"No nearby records found for {query_datetime}")
            return

        print(f"   Showing closest records (within ±2 hours):")
        matching_records = nearby_records

    # Display results
    print(f"\n{'='*80}")
    print(f"TRAFFIC QUERY RESULTS: {date_str} {time_str}")
    print(f"{'='*80}")

    for idx, row in matching_records.iterrows():
        dt = row['date_time']
        day_name = dt.strftime('%A')
        weather = row['weather_main']
        temp_celsius = row['temp'] - 273.15

        print(f"\nDate/Time:         {dt.strftime('%Y-%m-%d %H:%M')} ({day_name})")
        print(f"Traffic Volume:    {row['traffic_volume']:.0f} vehicles")
        print(f"Congestion Level:  {['Light', 'Moderate', 'Heavy', 'Very Heavy'][row['congestion_category']]}")
        print(f"Weather:           {weather}")
        print(f"Temperature:       {temp_celsius:.1f}°C ({row['temp']:.1f}K)")
        print(f"Cloud Coverage:    {row['clouds_all']:.0f}%")
        print(f"Rainfall (1h):     {row['rain_1h']:.2f}mm")

    logger.info(f"Successfully queried {len(matching_records)} record(s) for {query_datetime}")


def identify_peak_hours(df: pd.DataFrame, limit: int = 10) -> None:
    """
    Identify high-traffic periods.

    Args:
        df: Feature-engineered dataset
        limit: Number of top hours to display
    """
    logger.info(f"Command invoked: peak-hours with limit={limit}")

    print(f"\n{'='*80}")
    print(f"TOP {limit} HIGHEST TRAFFIC PERIODS")
    print(f"{'='*80}\n")

    # Calculate average traffic by hour
    hourly_stats = df.groupby('hour').agg({
        'traffic_volume': ['mean', 'std', 'min', 'max', 'count']
    }).round(0)
    hourly_stats.columns = ['mean', 'std', 'min', 'max', 'count']
    hourly_stats = hourly_stats.sort_values('mean', ascending=False)

    print(f"{'Hour':<6} {'Mean Volume':<15} {'Std Dev':<12} {'Min':<10} {'Max':<10} {'Records':<10}")
    print("-" * 80)

    for idx, (hour, row) in enumerate(hourly_stats.head(limit).iterrows(), 1):
        print(f"{hour:02d}:00  {row['mean']:>12.0f}   {row['std']:>10.0f}   "
              f"{row['min']:>8.0f}   {row['max']:>8.0f}   {row['count']:>8.0f}")

    # Peak hour overall
    peak_hour = hourly_stats.index[0]
    peak_volume = hourly_stats.iloc[0]['mean']
    logger.info(f"Peak hour identified: {peak_hour}:00 with average volume {peak_volume:.0f}")
    print(f"\n✓ Peak traffic hour: {peak_hour}:00 ({peak_volume:.0f} vehicles on average)")


def compare_weekday_weekend(df: pd.DataFrame) -> None:
    """
    Compare weekday and weekend traffic patterns.

    Args:
        df: Feature-engineered dataset
    """
    logger.info("Command invoked: weekday-compare")

    weekday_data = df[df['is_weekend'] == 0]
    weekend_data = df[df['is_weekend'] == 1]

    print(f"\n{'='*80}")
    print(f"WEEKDAY vs WEEKEND TRAFFIC COMPARISON")
    print(f"{'='*80}\n")

    # Overall statistics
    print("OVERALL STATISTICS:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Weekday':>20} {'Weekend':>20}")
    print("-" * 80)

    weekday_mean = weekday_data['traffic_volume'].mean()
    weekend_mean = weekend_data['traffic_volume'].mean()
    weekday_median = weekday_data['traffic_volume'].median()
    weekend_median = weekend_data['traffic_volume'].median()
    weekday_std = weekday_data['traffic_volume'].std()
    weekend_std = weekend_data['traffic_volume'].std()
    weekday_records = len(weekday_data)
    weekend_records = len(weekend_data)

    print(f"{'Mean Traffic Volume':<30} {weekday_mean:>20.0f} {weekend_mean:>20.0f}")
    print(f"{'Median Traffic Volume':<30} {weekday_median:>20.0f} {weekend_median:>20.0f}")
    print(f"{'Std Deviation':<30} {weekday_std:>20.0f} {weekend_std:>20.0f}")
    print(f"{'Total Records':<30} {weekday_records:>20.0f} {weekend_records:>20.0f}")

    diff_pct = (weekday_mean - weekend_mean) / weekend_mean * 100
    print(f"{'Difference':<30} {diff_pct:>19.1f}% {'higher on weekdays'}")

    # Peak hours for each
    print(f"\n{'PEAK HOURS BY DAY TYPE':<40}")
    print("-" * 80)

    weekday_hourly = weekday_data.groupby('hour')['traffic_volume'].mean().sort_values(ascending=False)
    weekend_hourly = weekend_data.groupby('hour')['traffic_volume'].mean().sort_values(ascending=False)

    print(f"{'Weekday Peak Hours':<40} {'Weekend Peak Hours':<40}")
    print("-" * 80)

    for i in range(5):
        wd_hour = weekday_hourly.index[i]
        wd_volume = weekday_hourly.iloc[i]
        we_hour = weekend_hourly.index[i]
        we_volume = weekend_hourly.iloc[i]

        print(f"{wd_hour:02d}:00 ({wd_volume:>6.0f} vehicles)    {we_hour:02d}:00 ({we_volume:>6.0f} vehicles)")

    logger.info(f"Weekday traffic is {diff_pct:.1f}% higher than weekends")


def recommend_travel_times(df: pd.DataFrame, max_volume: int = 2000) -> None:
    """
    Recommend best times to travel (lowest traffic periods).

    Args:
        df: Feature-engineered dataset
        max_volume: Maximum traffic volume threshold
    """
    logger.info(f"Command invoked: best-times with max_volume={max_volume}")

    print(f"\n{'='*80}")
    print(f"RECOMMENDED TRAVEL TIMES (Traffic Volume < {max_volume})")
    print(f"{'='*80}\n")

    # Find hours with low traffic
    low_traffic_hours = df[df['traffic_volume'] < max_volume].groupby('hour').size()
    low_traffic_hours = low_traffic_hours.sort_values(ascending=False)

    if low_traffic_hours.empty:
        print(f"⚠️  No hours with traffic below {max_volume} vehicles found.")
        logger.warning(f"No records found with traffic volume < {max_volume}")
        return

    print(f"{'Hour':<8} {'Records with Low Traffic':<30}")
    print("-" * 80)

    for hour, count in low_traffic_hours.head(10).items():
        pct = count / len(df) * 100
        print(f"{hour:02d}:00   {count:>6} records ({pct:>5.1f}%)")

    # Best day type for travel
    weekday_low = len(df[(df['is_weekend'] == 0) & (df['traffic_volume'] < max_volume)])
    weekend_low = len(df[(df['is_weekend'] == 1) & (df['traffic_volume'] < max_volume)])
    weekday_pct = weekday_low / len(df[df['is_weekend'] == 0]) * 100
    weekend_pct = weekend_low / len(df[df['is_weekend'] == 1]) * 100

    print(f"\n{'BEST DAY TYPES FOR LOW-TRAFFIC TRAVEL':<50}")
    print("-" * 80)
    print(f"Weekdays:  {weekday_pct:>5.1f}% of records have traffic < {max_volume}")
    print(f"Weekends:  {weekend_pct:>5.1f}% of records have traffic < {max_volume}")

    best_day_type = "Weekends" if weekend_pct > weekday_pct else "Weekdays"
    print(f"\n✓ Best day type: {best_day_type}")

    logger.info(f"Recommended travel times identified. Best day type: {best_day_type}")


def analyze_weather_impact(df: pd.DataFrame) -> None:
    """
    Analyze how different weather conditions affect traffic.

    Args:
        df: Feature-engineered dataset
    """
    logger.info("Command invoked: weather-impact")

    print(f"\n{'='*80}")
    print(f"WEATHER IMPACT ON TRAFFIC")
    print(f"{'='*80}\n")

    # Calculate statistics by weather
    weather_stats = df.groupby('weather_main').agg({
        'traffic_volume': ['mean', 'median', 'std', 'count']
    }).round(0)
    weather_stats.columns = ['mean', 'median', 'std', 'count']
    weather_stats = weather_stats.sort_values('mean', ascending=False)

    print(f"{'Weather Type':<20} {'Mean Volume':<15} {'Median':<12} {'Std Dev':<12} {'Records':<10}")
    print("-" * 80)

    for weather, row in weather_stats.iterrows():
        print(f"{weather:<20} {row['mean']:>12.0f}   {row['median']:>10.0f}   "
              f"{row['std']:>10.0f}   {row['count']:>8.0f}")

    # Best and worst weather for traffic
    best_weather = weather_stats.index[-1]
    worst_weather = weather_stats.index[0]
    best_volume = weather_stats.iloc[-1]['mean']
    worst_volume = weather_stats.iloc[0]['mean']
    diff_pct = (worst_volume - best_volume) / best_volume * 100

    print(f"\n{'='*80}")
    print(f"Best weather for traffic:   {best_weather} ({best_volume:.0f} vehicles)")
    print(f"Worst weather for traffic:  {worst_weather} ({worst_volume:.0f} vehicles)")
    print(f"Difference:                 {diff_pct:.1f}% higher in worst conditions")
    print(f"{'='*80}")

    logger.info(f"Best weather: {best_weather} ({best_volume:.0f}), Worst: {worst_weather} ({worst_volume:.0f})")


def interactive_menu(df: pd.DataFrame) -> None:
    """Display interactive menu for user to choose commands."""
    logger.info("Interactive menu mode activated")

    import os

    while True:
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')

        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  TRAFFIC ANALYTICS APPLICATION                             ║
║                   Metro Interstate Traffic Volume                          ║
╚════════════════════════════════════════════════════════════════════════════╝

SELECT AN OPTION:

1. Query Traffic by Date/Time
2. Find Peak Traffic Hours
3. Compare Weekday vs Weekend
4. Find Best Times to Travel
5. Analyze Weather Impact
6. View Help
0. Exit

════════════════════════════════════════════════════════════════════════════""")

        choice = input("\nEnter your choice (0-6): ").strip()

        if choice == "1":
            menu_query_time(df)
        elif choice == "2":
            menu_peak_hours(df)
        elif choice == "3":
            menu_weekday_compare(df)
        elif choice == "4":
            menu_best_times(df)
        elif choice == "5":
            menu_weather_impact(df)
        elif choice == "6":
            menu_help()
        elif choice == "0":
            logger.info("Application closed by user from menu")
            print("\n✓ Thank you for using Traffic Analytics Application!")
            print("  See traffic_app.log for detailed activity log.\n")
            break
        else:
            print("\n❌ Invalid choice. Please enter 0-6.")
            input("Press ENTER to continue...")


def menu_query_time(df: pd.DataFrame) -> None:
    """Interactive query by date and time."""
    logger.info("Menu option: Query by Date/Time")

    print("\n" + "="*80)
    print("QUERY TRAFFIC BY DATE AND TIME")
    print("="*80)

    while True:
        date_str = input("\nEnter date (YYYY-MM-DD) or press ENTER to go back: ").strip()
        if not date_str:
            return
        if len(date_str) == 10 and date_str.count('-') == 2:
            break
        print("❌ Invalid format. Use YYYY-MM-DD")

    while True:
        time_str = input("Enter time (HH:MM) in 24-hour format: ").strip()
        if len(time_str) == 5 and time_str.count(':') == 1:
            break
        print("❌ Invalid format. Use HH:MM")

    query_by_time(df, date_str, time_str)
    input("\nPress ENTER to continue...")


def menu_peak_hours(df: pd.DataFrame) -> None:
    """Interactive peak hours display."""
    logger.info("Menu option: Peak Hours")

    while True:
        try:
            limit_str = input("\nHow many peak hours to show? (default: 10): ").strip()
            limit = int(limit_str) if limit_str else 10
            if limit > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

    identify_peak_hours(df, limit)
    input("\nPress ENTER to continue...")


def menu_weekday_compare(df: pd.DataFrame) -> None:
    """Interactive weekday vs weekend comparison."""
    logger.info("Menu option: Weekday Compare")

    compare_weekday_weekend(df)
    input("\nPress ENTER to continue...")


def menu_best_times(df: pd.DataFrame) -> None:
    """Interactive best times recommendation."""
    logger.info("Menu option: Best Times")

    while True:
        try:
            max_vol_str = input("\nMaximum traffic volume threshold (default: 2000): ").strip()
            max_volume = int(max_vol_str) if max_vol_str else 2000
            if max_volume > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

    recommend_travel_times(df, max_volume)
    input("\nPress ENTER to continue...")


def menu_weather_impact(df: pd.DataFrame) -> None:
    """Interactive weather impact analysis."""
    logger.info("Menu option: Weather Impact")

    analyze_weather_impact(df)
    input("\nPress ENTER to continue...")


def menu_help() -> None:
    """Interactive help display."""
    import os

    logger.info("Help accessed from menu")

    os.system('cls' if os.name == 'nt' else 'clear')

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    COMMAND DESCRIPTIONS                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

1. QUERY TRAFFIC BY DATE/TIME
   Find traffic information for a specific date and time.
   Returns: Volume, weather, temperature, congestion level

2. FIND PEAK TRAFFIC HOURS
   Shows the highest traffic periods with statistics.
   Returns: Top hours ranked by average traffic volume

3. COMPARE WEEKDAY VS WEEKEND
   Compare traffic patterns between weekdays and weekends.
   Returns: Statistics and peak hours for each day type

4. FIND BEST TIMES TO TRAVEL
   Recommend lowest traffic times to travel.
   Returns: Hours and days with traffic below threshold

5. ANALYZE WEATHER IMPACT
   See how weather conditions affect traffic.
   Returns: Traffic volume by weather type

════════════════════════════════════════════════════════════════════════════

KEY INSIGHTS:
  ✓ Peak traffic: 16:00 (4 PM) with ~5,664 vehicles
  ✓ Weekday traffic: 37.4% higher than weekends
  ✓ Best travel time: 04:00 AM (lowest traffic)
  ✓ Data period: 2012-10-02 to 2018-09-30

════════════════════════════════════════════════════════════════════════════
""")

    input("Press ENTER to return to menu...")


def query_by_time(df: pd.DataFrame, date_str: str, time_str: str) -> None:
    """Query traffic for a specific date and time."""
    logger.info(f"Command invoked: query-time with date={date_str}, time={time_str}")

    try:
        query_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        logger.debug(f"Parsed datetime: {query_datetime}")
    except ValueError:
        error_msg = f"Invalid date/time format. Use YYYY-MM-DD for date and HH:MM for time."
        logger.error(f"User input error: {error_msg}")
        print(f"\n❌ Error: {error_msg}")
        return

    hour = query_datetime.hour
    query_date = query_datetime.date()

    matching_records = df[
        (df['date_time'].dt.hour == hour) &
        (df['date_time'].dt.date == query_date)
    ]

    if matching_records.empty:
        logger.warning(f"No exact match found for {query_datetime}")
        print(f"\n⚠️  No records found for {date_str} {time_str}")

        nearby_records = df[
            (df['date_time'].dt.date == query_date) &
            (df['date_time'].dt.hour.isin(range(max(0, hour - 2), min(24, hour + 3))))
        ]

        if nearby_records.empty:
            print("   No nearby records within ±2 hours on the same date.")
            logger.info(f"No nearby records found for {query_datetime}")
            return

        print(f"   Showing closest records (within ±2 hours):")
        matching_records = nearby_records

    print(f"\n{'='*80}")
    print(f"TRAFFIC QUERY RESULTS: {date_str} {time_str}")
    print(f"{'='*80}")

    for idx, row in matching_records.iterrows():
        dt = row['date_time']
        day_name = dt.strftime('%A')
        weather = row['weather_main']
        temp_celsius = row['temp'] - 273.15

        print(f"\nDate/Time:         {dt.strftime('%Y-%m-%d %H:%M')} ({day_name})")
        print(f"Traffic Volume:    {row['traffic_volume']:.0f} vehicles")
        print(f"Congestion Level:  {['Light', 'Moderate', 'Heavy', 'Very Heavy'][row['congestion_category']]}")
        print(f"Weather:           {weather}")
        print(f"Temperature:       {temp_celsius:.1f}°C ({row['temp']:.1f}K)")
        print(f"Cloud Coverage:    {row['clouds_all']:.0f}%")
        print(f"Rainfall (1h):     {row['rain_1h']:.2f}mm")

    logger.info(f"Successfully queried {len(matching_records)} record(s) for {query_datetime}")


def main():
    """Main application entry point."""
    configure_logging()

    logger.info("=" * 80)
    logger.info("Traffic Analytics Application Started")
    logger.info("=" * 80)

    # Load dataset
    df = load_dataset("Metro_Interstate_Traffic_Volume_features.csv")
    if df is None:
        print("❌ Failed to load dataset. Cannot proceed.")
        logger.error("Dataset loading failed. Application exiting.")
        sys.exit(1)

    # Check command-line arguments
    if len(sys.argv) < 2:
        # No arguments provided - show interactive menu
        logger.info("No command-line arguments. Starting interactive menu.")
        interactive_menu(df)
        return

    command = sys.argv[1].lower()

    # Route commands
    try:
        if command == "help":
            display_help()

        elif command == "query-time":
            if len(sys.argv) < 4:
                error_msg = "query-time requires both date and time arguments"
                logger.error(f"User input error: {error_msg}")
                print(f"\n❌ Error: {error_msg}")
                print("Usage: python traffic_analytics_app.py query-time <YYYY-MM-DD> <HH:MM>\n")
                sys.exit(1)

            date_str = sys.argv[2]
            time_str = sys.argv[3]
            query_by_time(df, date_str, time_str)

        elif command == "peak-hours":
            limit = 10
            if "--limit" in sys.argv:
                try:
                    limit_idx = sys.argv.index("--limit")
                    limit = int(sys.argv[limit_idx + 1])
                except (IndexError, ValueError):
                    logger.error("Invalid --limit value. Using default.")
                    print("⚠️  Invalid --limit value. Using default (10).")

            identify_peak_hours(df, limit)

        elif command == "weekday-compare":
            compare_weekday_weekend(df)

        elif command == "best-times":
            max_volume = 2000
            if "--max-volume" in sys.argv:
                try:
                    vol_idx = sys.argv.index("--max-volume")
                    max_volume = int(sys.argv[vol_idx + 1])
                except (IndexError, ValueError):
                    logger.error("Invalid --max-volume value. Using default.")
                    print("⚠️  Invalid --max-volume value. Using default (2000).")

            recommend_travel_times(df, max_volume)

        elif command == "weather-impact":
            analyze_weather_impact(df)

        else:
            error_msg = f"Unknown command: {command}. Use 'help' to see available commands."
            logger.error(f"User input error: {error_msg}")
            print(f"\n❌ Error: {error_msg}\n")
            sys.exit(1)

        logger.info(f"Command '{command}' completed successfully")
        print("\n" + "=" * 80)

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.error(error_msg, exc_info=False)
        print(f"\n❌ Error: {error_msg}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
