import boto3
import json
from botocore.exceptions import ClientError

# Prompt
prompt_data = "Act as Shakespeare and write a poem on country Nepal"

# Format using LLaMA 3 native chat instruction format
formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt_data}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")  # use your region

# Payload for LLaMA 3
payload = {
    "prompt": formatted_prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

try:
    response = bedrock.invoke_model(
        body=json.dumps(payload),
        modelId="meta.llama3-70b-instruct-v1:0",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response["body"].read())
    print(response_body["generation"])

except ClientError as e:
    print("Access or invocation error:", e.response["Error"]["Message"])
except Exception as e:
    print("Unexpected error:", str(e))
