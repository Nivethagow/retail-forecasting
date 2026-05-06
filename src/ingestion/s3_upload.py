"""
src/ingestion/s3_upload.py
--------------------------
Upload the raw Kaggle CSV to S3 and verify the Glue crawler
is configured to scan it.

Usage:
    python src/ingestion/s3_upload.py \
        --file data/raw/retail_store_inventory.csv \
        --config configs/aws_config.yaml

Prerequisites:
    pip install boto3 pyyaml
    AWS credentials configured via ~/.aws/credentials or environment variables
"""

import argparse
import os
import sys
import yaml
import boto3
from pathlib import Path
from botocore.exceptions import ClientError


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def upload_file_to_s3(local_path: str, bucket: str, s3_key: str, region: str) -> bool:
    """Upload a local file to S3. Returns True on success."""
    s3 = boto3.client("s3", region_name=region)
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Uploading {local_path} ({file_size_mb:.1f} MB) → s3://{bucket}/{s3_key}")
    try:
        s3.upload_file(local_path, bucket, s3_key)
        print(f"  ✓ Upload complete")
        return True
    except ClientError as e:
        print(f"  ✗ Upload failed: {e}")
        return False


def ensure_bucket_exists(bucket: str, region: str) -> None:
    """Create the S3 bucket if it does not already exist."""
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  ✓ Bucket s3://{bucket} already exists")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"  Creating bucket s3://{bucket} in {region}...")
            if region == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
            print(f"  ✓ Bucket created")
        else:
            raise


def start_glue_crawler(crawler_name: str, region: str) -> None:
    """Trigger the Glue crawler to refresh the Data Catalog."""
    glue = boto3.client("glue", region_name=region)
    try:
        glue.start_crawler(Name=crawler_name)
        print(f"  ✓ Glue crawler '{crawler_name}' started")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "CrawlerRunningException":
            print(f"  ℹ Crawler '{crawler_name}' is already running")
        elif code == "EntityNotFoundException":
            print(f"  ℹ Crawler '{crawler_name}' not found — create it in the AWS Console first")
        else:
            raise


def main():
    parser = argparse.ArgumentParser(description="Upload retail CSV to S3 and trigger Glue crawler")
    parser.add_argument("--file", required=True, help="Path to the local CSV file")
    parser.add_argument("--config", default="configs/aws_config.yaml", help="AWS config YAML")
    parser.add_argument("--skip-crawler", action="store_true", help="Skip triggering the Glue crawler")
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    cfg = load_config(args.config)
    region = cfg["aws"]["region"]
    bucket = cfg["s3"]["bucket"]
    raw_prefix = cfg["s3"]["prefixes"]["raw"]
    crawler_name = cfg["glue"]["crawler_name"]

    filename = Path(args.file).name
    s3_key = raw_prefix + filename

    print("\n── Step 1: Ensure S3 bucket exists ──")
    ensure_bucket_exists(bucket, region)

    print("\n── Step 2: Upload CSV ──")
    success = upload_file_to_s3(args.file, bucket, s3_key, region)
    if not success:
        sys.exit(1)

    if not args.skip_crawler:
        print("\n── Step 3: Start Glue crawler ──")
        start_glue_crawler(crawler_name, region)

    print(f"\n✓ Done. Data available at: s3://{bucket}/{s3_key}")
    print(f"  Next step: Run the Glue crawler in the AWS Console, then open Athena to verify the schema.")


if __name__ == "__main__":
    main()
