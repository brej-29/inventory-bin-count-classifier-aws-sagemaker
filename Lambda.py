import json
import logging
import os
import boto3
import urllib.request
from urllib.parse import urlparse
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm_runtime = boto3.client("sagemaker-runtime")
s3 = boto3.client("s3")

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "").strip()

def _parse_event(event):
    # Supports both direct JSON dict and API Gateway proxy events
    if isinstance(event, dict) and "body" in event and isinstance(event["body"], str):
        try:
            body = json.loads(event["body"])
            return body if isinstance(body, dict) else {}
        except Exception:
            return {}
    return event if isinstance(event, dict) else {}

def _download_http_image(url: str) -> bytes:
    logger.info(f"Downloading image from URL: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "lambda-sagemaker-invoker/1.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP download failed: {resp.status}")
        return resp.read()

def _download_s3_image(s3_uri: str) -> bytes:
    logger.info(f"Downloading image from S3 URI: {s3_uri}")
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3_uri: {s3_uri}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def lambda_handler(event, context):
    try:
        if not ENDPOINT_NAME:
            raise ValueError("ENDPOINT_NAME is not set. Add it as a Lambda environment variable.")

        payload = _parse_event(event)
        image_url = payload.get("image_url")
        s3_uri = payload.get("s3_uri")

        if not image_url and not s3_uri:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({
                    "error": "Missing input. Provide either 'image_url' or 's3_uri'."
                })
            }

        # 1) Fetch image bytes
        if image_url:
            img_bytes = _download_http_image(image_url)
            input_meta = {"image_url": image_url}
        else:
            img_bytes = _download_s3_image(s3_uri)
            input_meta = {"s3_uri": s3_uri}

        # 2) Invoke SageMaker endpoint with raw bytes (matches your inference content-type)
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/x-image",
            Accept="application/json",
            Body=img_bytes
        )

        result_str = response["Body"].read().decode("utf-8")
        result_json = json.loads(result_str)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "endpoint": ENDPOINT_NAME,
                "input": input_meta,
                "result": result_json
            })
        }

    except urllib.error.URLError as e:
        logger.exception("HTTP download error")
        return {
            "statusCode": 502,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Failed to download image from URL", "details": str(e)})
        }

    except ClientError as e:
        logger.exception("AWS ClientError")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "AWS ClientError", "details": str(e)})
        }

    except Exception as e:
        logger.exception("Unhandled error")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Unhandled exception", "details": str(e)})
        }