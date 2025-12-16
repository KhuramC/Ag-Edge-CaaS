import json
import boto3
import base64
from datetime import datetime
import traceback

s3_client = boto3.client('s3')

def parse_multipart(body, boundary):
    """Parse multipart/form-data to extract file"""
    parts = body.split(boundary)
    
    for part in parts:
        if b'Content-Disposition' in part and b'filename=' in part:
            # Extract filename
            filename_start = part.find(b'filename="') + 10
            filename_end = part.find(b'"', filename_start)
            filename = part[filename_start:filename_end].decode('utf-8')
            
            # Extract file content (after double CRLF)
            content_start = part.find(b'\r\n\r\n') + 4
            content_end = part.rfind(b'\r\n')
            content = part[content_start:content_end]
            
            return filename, content
    
    return None, None

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event, default=str)}")
    
    bucket_name = 'cloud-computing-drone-data-bucket'
    
    try:
        # Get Content-Type header
        headers = event.get('headers', {})
        content_type = headers.get('Content-Type') or headers.get('content-type', '')
        
        # Validate Content-Type
        if 'multipart/form-data' not in content_type:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': 'Invalid Content-Type',
                    'message': 'Must use multipart/form-data for file uploads',
                    'received': content_type
                })
            }
        
        # Extract boundary from Content-Type
        if 'boundary=' not in content_type:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': 'Missing boundary in Content-Type header'
                })
            }
        
        boundary = content_type.split('boundary=')[1]
        boundary = ('--' + boundary).encode()
        
        # Decode body (API Gateway base64 encodes multipart data)
        if event.get('isBase64Encoded', False):
            body = base64.b64decode(event['body'])
        else:
            # Fallback if not base64 encoded
            body = event['body'].encode('utf-8') if isinstance(event['body'], str) else event['body']
        
        # Parse multipart data
        filename, csv_content = parse_multipart(body, boundary)
        
        # Validate file was found
        if not filename or not csv_content:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': 'No file found in request',
                    'message': 'Please include a file in the multipart/form-data request'
                })
            }
        
        # Validate file extension
        if not filename.lower().endswith('.csv'):
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': 'Invalid file type',
                    'message': 'Only .csv files are accepted',
                    'received_file': filename
                })
            }
        
        # Decode CSV content to text
        try:
            csv_text = csv_content.decode('utf-8')
        except UnicodeDecodeError:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': 'Invalid file encoding',
                    'message': 'CSV file must be UTF-8 encoded'
                })
            }
        
        # Validate CSV is not empty
        if not csv_text.strip():
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': 'Empty CSV file',
                    'message': 'CSV file cannot be empty'
                })
            }
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        s3_filename = f'{timestamp}.csv'
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_filename,
            Body=csv_text,
            ContentType='text/csv',
            Metadata={
                'original-filename': filename,
                'upload-timestamp': timestamp
            }
        )
        
        print(f"Successfully uploaded {s3_filename} to {bucket_name}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'CSV file uploaded successfully',
                'original_filename': filename,
                's3_filename': s3_filename,
                'bucket': bucket_name,
                'timestamp': timestamp
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }