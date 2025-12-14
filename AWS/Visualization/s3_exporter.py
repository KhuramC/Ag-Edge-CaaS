import boto3
import csv
from io import StringIO
from flask import Flask, Response
from datetime import datetime, time as dt_time
import pytz
import os

# --- CONFIGURATION ---
S3_BUCKET = os.environ.get("S3_BUCKET", "cloud-computing-drone-output-bucket")
EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", 9999))
CENTRAL_TIME_ZONE = pytz.timezone('America/Chicago')
UTC_TIME_ZONE = pytz.utc
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

# CSV columns being tracked as metrics
METRIC_COLUMNS = [
    'channel', 'snr_bs1', 'snr_bs2', 'snr_bs3', 'snr_bs4', 'data_total_mb',
    'data_downloaded_mb', 'data_remaining_mb', 'throughput_mbps', 'latency_ms',
    'rtt_ms', 'delay_ms', 'distance_to_selected_bs_m', 'waypoint_index',
    'num_waypoints', 'lat', 'lon', 'alt', 'elapsed_time_s', 'idqn_freq_ghz'
]

# CSV columns being used as labels
LABEL_COLUMNS = [
    'drone_id',
]
# could add "bs_id_selected" in the future

app = Flask(__name__)
s3_client = boto3.client('s3')

def get_all_csv_files() -> list:
    """Returns a list of keys for ALL CSV files in the bucket."""
    objects = s3_client.list_objects_v2(Bucket=S3_BUCKET)
    if not objects.get('Contents'):
        return []
    return [obj['Key'] for obj in objects['Contents'] if obj['Key'].lower().endswith('.csv')]

def parse_iso_to_epoch_ms(iso_string):
    """Converts ISO 8601 string to millisecond epoch time."""
    # datetime.strptime handles the parsing
    dt_obj = datetime.strptime(iso_string, TIME_FORMAT)
    
    # We must localize to UTC because the string doesn't contain timezone info (the 'Z' or offset is missing)
    dt_utc = UTC_TIME_ZONE.localize(dt_obj)
    
    return int(dt_utc.timestamp() * 1000)
        
def calculate_midnight_ct_ms():
    """Calculates Midnight (00:00:00) Central Time (CT) for the current day."""
    
    CENTRAL_TIME_ZONE = pytz.timezone('America/Chicago')
    now_ct = datetime.now(CENTRAL_TIME_ZONE).date()
    
    # Combine with Midnight (00:00:00)
    base_time_dt_ct = CENTRAL_TIME_ZONE.localize(
        datetime.combine(now_ct, dt_time(0, 0, 0))
    )
    
    base_time_epoch_ms = int(base_time_dt_ct.timestamp() * 1000)
    
    return base_time_epoch_ms

def generate_prometheus_metrics():
    """Generates Prometheus time series based on the available CSV files in the 
    S3 bucket."""
    
    
    csv_keys = get_all_csv_files()

    if not csv_keys:
        yield "# No CSV files found in S3 bucket.\n"
        return
    
    BASE_TIME_MS = calculate_midnight_ct_ms()

    for col in METRIC_COLUMNS:
        # Prometheus metric names should be snake_case
        metric_name = col.replace('_', '_') 
        yield f'# HELP {metric_name} Metric derived from CSV column {col}.\n'
        yield f'# TYPE {metric_name} gauge\n'

    for key in csv_keys:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            csv_content = response['Body'].read().decode('utf-8')

            csv_file = StringIO(csv_content)
            csv_reader = csv.DictReader(csv_file)

            filename = key.split('/')[-1].replace('.csv', '')

            # Base labels for full csv file
            base_labels = f'source="{filename}",bucket="{S3_BUCKET}"'
            # Time the experiment starts.
            exp_start_ms = None

            for i, row in enumerate(csv_reader):
                try:
                    # --- TIME NORMALIZATION LOGIC ---
                    if 'timestamp' in row:
                        current_timestamp_ms = parse_iso_to_epoch_ms(row["timestamp"])
                    else:
                        current_timestamp_ms = i * 1000

                    if exp_start_ms is None:
                        exp_start_ms = current_timestamp_ms
                    delta_ms = current_timestamp_ms - exp_start_ms
                    final_timestamp = int(BASE_TIME_MS + delta_ms)

                    # --- EXTRACT DATA ---
                    labels = f'{base_labels}'
                    # add row specific labels
                    for label_col in LABEL_COLUMNS:
                        if label_col in row:
                            labels += f',{label_col}="{row[label_col]}"'

                    for metric_col in METRIC_COLUMNS:
                        metric_value = row.get(metric_col)
                        
                        if metric_value is not None:
                            # Ensure value is a float
                            metric_value = float(metric_value)
                            
                            # Use the column name as the Prometheus metric name
                            metric_name = metric_col.replace('_', '_')
                            
                            yield f'{metric_name}{{{labels}}} {metric_value} {final_timestamp}\n'

                except (ValueError, KeyError) as e:
                    print(f"Skipping row {i} due to error: {e} in file {key}.")
                    continue

        except Exception as e:
            yield f'# Error reading file {key}: {str(e)}\n'

@app.route('/metrics')
def metrics():
    return Response(generate_prometheus_metrics(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=EXPORTER_PORT)
