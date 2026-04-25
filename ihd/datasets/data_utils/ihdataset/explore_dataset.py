import boto3
from botocore import UNSIGNED
from botocore.client import Config

# Set up anonymous access
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = 'ihdataset-01'

# List top-level directories
response = s3.list_objects_v2(Bucket=bucket, Delimiter='/')
for prefix in response.get('CommonPrefixes', []):
    print("Top-level directory:", prefix['Prefix'])