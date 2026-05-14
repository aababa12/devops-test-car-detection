import logging
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


logger = logging.getLogger(__name__)


class S3DownloadError(RuntimeError):
    """Raised when the source video cannot be downloaded from S3."""


class S3VideoDownloader:
    def __init__(self, region_name: str) -> None:
        self._client = boto3.client("s3", region_name=region_name)

    def download(self, bucket: str, key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s to %s", bucket, key, destination)

        try:
            self._client.download_file(bucket, key, str(destination))
        except (BotoCoreError, ClientError) as exc:
            raise S3DownloadError(f"Failed to download s3://{bucket}/{key}") from exc

        return destination
