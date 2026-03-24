import boto3
import os

s3=boto3.client('s3',region_name='ap-northeast-1')
# response=s3.list_buckets()
# for bucket in response['Buckets']:
#     print(bucket['Name'])
BUCKET_NAME='ml-training-breast-cancer-data'
local_root='datasets'


# Create a reusable Paginator
paginator = s3.get_paginator('list_objects_v2')

import boto3

s3 = boto3.client("s3")

def get_latest_augmented_prefix():
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix="augmented/",
        Delimiter="/"
    )

    folders = [p["Prefix"] for p in response.get("CommonPrefixes", [])]

    latest = sorted(folders)[-1]

    print("Using latest:", latest)
    return latest


def download_from_s3(prefix, local_root):
    """
    Download all files from an S3 prefix into a local directory.
    
    Args:
        bucket_name (str): S3 bucket name
        prefix (str): S3 folder path (e.g., 'original/train/')
        local_root (str): Local directory (e.g., './datasets/train')
    """

    # Step 1: Create local root directory if it doesn't exist
    os.makedirs(local_root, exist_ok=True)

    # Step 2: Create paginator to handle large number of files
    paginator = s3.get_paginator("list_objects_v2")

    # Step 3: Iterate through all pages (S3 returns max 1000 objects per page)
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):

        # Step 4: Get list of files in this page
        for obj in page.get("Contents", []):

            # Step 5: Get file key (path in S3)
            key = obj["Key"]
            print(f"Processing S3 key: {key}")
            
            # Step 6: Skip folders (S3 sometimes includes them)
            if key.endswith("/"):
                continue

            # Step 7: Remove prefix → get relative path
            relative_path = os.path.relpath(key, prefix)
            print(f"Relative path: {relative_path}")
            # Step 8: Create full local path
            local_file_path = os.path.join(local_root, relative_path)
            print(f"Local file path: {local_file_path}")

            # Step 9: Create directories if needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            print(f"Ensured directory exists: {os.path.dirname(local_file_path)}")


            # Step 10: Skip download if file already exists (caching)
            if os.path.exists(local_file_path):
                print(f"⏭️ Skipping (already exists): {local_file_path}")
                continue

            # Step 11: Download file from S3
            s3.download_file(BUCKET_NAME, key, local_file_path)

            print(f"Downloaded: {key} → {local_file_path}")


# Processing S3 key: original/train/benign/benign (181)_mask.png
# Relative path: benign\benign (181)_mask.png
# Local file path: ./data/original/train\benign\benign (181)_mask.png
# Ensured directory exists: ./data/original/train\benign
# Downloaded: original/train/benign/benign (181)_mask.png → ./data/original/train\benign\benign (181)_mask.png
