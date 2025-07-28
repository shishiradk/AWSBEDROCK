import boto3
import json
from botocore.exceptions import ClientError

# Prompt
prompt_data = "Act as Shakespeare and write a poem on Gautam Buddha"

# Create Bedrock Runtime client (ensure region is supported)
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Native request format for Claude 3
payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.8,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_data}]
        }
    ]
}

# Convert to JSON
body = json.dumps(payload)
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )

    # Decode response body
    response_body = json.loads(response["body"].read())
    response_text = response_body["content"][0]["text"]
    print(response_text)

except ClientError as e:
    print("ClientError:", e.response["Error"]["Message"])
except Exception as e:
    print("Unexpected error:", str(e))
