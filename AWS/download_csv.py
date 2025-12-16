import boto3

bucket_name = 'cloud-computing-drone-data-bucket'


def get_most_recent_csv():
    """Get the most recent CSV file from S3 based on 'LastModified' timestamp."""
    s3_client = boto3.client('s3')
    
    try:
        # List all objects in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' not in response:
            print("No files found in bucket")
            return None
        
        # Filter only CSV files
        csv_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found in bucket")
            return None
        
        # Sort by LastModified (most recent first)
        most_recent = max(csv_files, key=lambda x: x['LastModified'])
        
        print(f"Most recent CSV: {most_recent['Key']}")
        print(f"Last Modified: {most_recent['LastModified']}")
        print(f"Size: {most_recent['Size']} bytes")
        
        return most_recent['Key']
        
    except Exception as e:
        print(f"Error listing S3 objects: {str(e)}")
        return None

def download_csv(file_key:str="training_data"):
    """Download CSV file from S3 to /tmp directory"""
    s3_client = boto3.client('s3')

    # Create local filepath in /tmp
    local_file = f'/tmp/{file_key}'

    try:
        print(f"Downloading {file_key} to {local_file}...")
        s3_client.download_file(bucket_name, file_key, local_file)
        print(f"Download successful: {local_file}")
        return local_file

    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return None
