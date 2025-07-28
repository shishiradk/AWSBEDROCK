import boto3
import json
import base64
import os
import random

# 1. Define your prompt
prompt_data = """
Provide me a 4K HD image of a beach, with a blue sky, rainy season feel, and cinematic display.
"""
prompt_template = [{"text": prompt_data}]

# 2. Create Bedrock client (use supported region)
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# 3. Prepare payload in native format
payload = {
    "text_prompts": prompt_template,
    "cfg_scale": 10,
    "seed": random.randint(0, 4294967295),  # optional but realistic
    "steps": 50,
    "width": 1024,   # Higher resolution for 4K-ish detail
    "height": 1024,
    "style_preset": "photographic"  # Recommended for real-world scenes
}

# 4. Convert to JSON and call model
body = json.dumps(payload)
model_id = "stability.stable-diffusion-xl-v1"  # v1 is latest/stable

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

# 5. Extract and decode image
response_body = json.loads(response["body"].read())
artifact = response_body["artifacts"][0]
image_base64 = artifact["base64"]
image_bytes = base64.b64decode(image_base64)

# 6. Save image to file
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, "generated-image.png")

with open(file_path, "wb") as f:
    f.write(image_bytes)

print(f" Image saved at: {file_path}")
