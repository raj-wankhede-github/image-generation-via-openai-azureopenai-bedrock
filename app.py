from flask import Flask, jsonify, request
import requests
import os, json, time, tiktoken
from io import BytesIO
from openai import AzureOpenAI
from openai import OpenAI

import boto3
import base64
import re

try:
    import awsgi
except ImportError:
    awsgi = None

OPENAI_AZURE_SIZE_VALUES = ["1024x1024", "1792x1024", "1024x1792"]
OPENAI_AZURE_QUALITY_VALUES = ["standard", "hd"]
BEDROCK_QUALITY_VALUES = ["standard", "premium"]
BEDROCK_SIZE_VALUES = [
    "1024x1024",
    "768x768",
    "512x512",
    "768x1152",
    "384x576",
    "1152x768",
    "576x384",
    "768x1280",
    "384x640",
    "1280x768",
    "640x384",
    "896x1152",
    "448x576",
    "1152x896",
    "576x448",
    "768x1408",
    "384x704",
    "1408x768",
    "704x384",
    "640x1408",
    "320x704",
    "1408x640",
    "704x320",
    "1152x640",
    "1173x640",
]

# This initializes the clients Bedrock Runtime and S3
bedrock_runtime_client = boto3.client('bedrock-runtime', region_name='us-east-1')
s3 = boto3.client('s3')
bucket_name = os.getenv("S3_BUCKET_NAME")
Client = None

app = Flask(__name__)
app.config["TIMEOUT"] = 60  # sets the timeout limit to 60 seconds


def error_response(message, status_code, details=None):
    payload = {"error": message}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status_code


def slugify_value(value, max_length=80):
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "na"
    return value[:max_length]


def build_object_key(api_type, user_id, deployment_id, prompt_text):
    provider = slugify_value(api_type, 24)
    user_part = slugify_value(user_id, 48)
    deployment_part = slugify_value(deployment_id, 48)
    prompt_part = slugify_value(prompt_text, 96)
    timestamp = int(time.time())
    return f"generated_images/{provider}/{user_part}_{deployment_part}_{timestamp}_{prompt_part}.jpg"

def generate_using_bedrock_and_upload_obj_to_s3(model_name, prompt, size, seed, quality, object_key):
    try:
        ## user pass input as width x height
        width = int(size.split("x")[0])
        height = int(size.split("x")[1])

        request = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": prompt},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": quality,
                "cfgScale": 8.0,
                # "width": 1408,
                # "height": 640, 
                "width": width,
                "height": height,
                "seed": seed,
            },
        })

        response = bedrock_runtime_client.invoke_model(
            modelId=model_name, body=request
        )

        response_body = json.loads(response["body"].read())
        base64_image_data = response_body["images"][0]

        image_data = base64.b64decode(base64_image_data)
        
        
        # Now we upload the image data to S3
        s3.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=image_data,
            ContentType='image/jpeg' 
        )
        image_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
        return image_url, "SUCCESS"
        
    except Exception as e:
        print(f"Exception: {str(e)}")
        print(f"Couldn't invoke Titan Image generator or Upload to S3 operation failed: {str(e)}")
        return str(e), "ERROR"


def generate_and_upload_obj_to_s3(model_name, prompt, size, quality, object_key ):

    try:
        response = Client.images.generate(
          model=model_name,
          prompt = prompt,
          size=size,
          quality=quality,
          n=1,
        )
        print(response)

        print("requesting image from URL")
        image_url = response.data[0].url
        response = requests.get(image_url, stream=True) 
    
        time.sleep(3)
        # Check for successful download
        if response.status_code == 200:
             
            # Create in-memory buffer for image data
            image_data = BytesIO()
            
            # Write downloaded data to buffer
            for chunk in response.iter_content(1024):
              image_data.write(chunk)
            
            # Reset buffer position
            image_data.seek(0)
            
            print("Uploading file to S3 bucket.")
            s3.upload_fileobj(image_data, bucket_name, object_key)
            print("S3 upload obj successful")
            image_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
            return image_url, "SUCCESS"

        else:
            image_url = None
            return image_url, "ERROR"

    except Exception as e:
        print(str(e))
        if "Error code: 400" in str(e):
            return str(e), "ERROR"
        else: 
            return str(e), "ERROR"

        
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("p50k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
def get_api_type(api_type_by_user):
    if api_type_by_user.startswith("azure"):
        api_type_by_user = "azure"
    elif api_type_by_user.startswith("open"):
        api_type_by_user = "open ai"
    elif api_type_by_user.startswith("bedrock"):
        api_type_by_user = "bedrock"
    
    return api_type_by_user


def validate_prompt_payload(form, query):
    missing_form_fields = [
        field for field in ["user_id", "deployment_id", "prompt"] if not form.get(field)
    ]
    missing_query_fields = [
        field for field in ["API-TYPE", "Quality", "Size"] if not query.get(field)
    ]

    errors = []
    if missing_form_fields:
        errors.append(f"Missing form fields: {', '.join(missing_form_fields)}")
    if missing_query_fields:
        errors.append(f"Missing query parameters: {', '.join(missing_query_fields)}")

    if errors:
        return {"ok": False, "errors": errors}

    prompt_text = form.get("prompt", "").strip()
    if not prompt_text:
        return {"ok": False, "errors": ["'prompt' must not be empty."]}

    api_type_input = query.get("API-TYPE", "").strip().lower()
    api_type = get_api_type(api_type_input)
    if api_type not in ["azure", "open ai", "bedrock"]:
        return {
            "ok": False,
            "errors": ["Invalid API-TYPE. Use one of: Azure Open AI, Open AI, Bedrock."],
        }

    quality = query.get("Quality", "").strip().lower()
    size = query.get("Size", "").strip()

    if api_type in ["azure", "open ai"]:
        if size not in OPENAI_AZURE_SIZE_VALUES:
            return {
                "ok": False,
                "errors": [
                    f"Invalid Size for OpenAI/AzureOpenAI: '{size}'. Allowed: {OPENAI_AZURE_SIZE_VALUES}"
                ],
            }
        if quality not in OPENAI_AZURE_QUALITY_VALUES:
            return {
                "ok": False,
                "errors": [
                    f"Invalid Quality for OpenAI/AzureOpenAI: '{quality}'. Allowed: {OPENAI_AZURE_QUALITY_VALUES}"
                ],
            }
    else:
        if size not in BEDROCK_SIZE_VALUES:
            return {
                "ok": False,
                "errors": [f"Invalid Size for Bedrock: '{size}'. Allowed: {BEDROCK_SIZE_VALUES}"],
            }
        if quality not in BEDROCK_QUALITY_VALUES:
            return {
                "ok": False,
                "errors": [
                    f"Invalid Quality for Bedrock: '{quality}'. Allowed: {BEDROCK_QUALITY_VALUES}"
                ],
            }

    return {
        "ok": True,
        "api_type": api_type,
        "quality": quality,
        "size": size,
        "prompt": prompt_text,
        "user_id": form.get("user_id"),
        "deployment_id": form.get("deployment_id"),
    }


def initialize_client(api_type_by_user, api_key=None):
    global Client

    if api_type_by_user == "azure":
        chosen_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not chosen_api_key or not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required for Azure OpenAI.")
        Client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=chosen_api_key,
            api_version="2024-02-01",
        )
        return

    if api_type_by_user == "open ai":
        if api_key:
            Client = OpenAI(api_key=api_key)
        else:
            Client = OpenAI()
        return

    # Bedrock does not require OpenAI client initialization.
    Client = None
   

def remove_pdfs(user_id, deployment_id):
    # Get the Pinecone Index object for the target namespace
    sp_name = f"namespace_{user_id}_{deployment_id}"
    # Delete all vectors from the target namespace
    print(f"Deleting: {sp_name}")
    delete_response = pinecone_index.delete(delete_all=True, namespace=sp_name)
    return delete_response


@app.route("/prompt", methods=["POST"])
def prompt():
    print("prompt")
    validation = validate_prompt_payload(request.form, request.args)
    if not validation["ok"]:
        return error_response("Request validation failed.", 400, validation["errors"])

    if not bucket_name:
        return error_response("Server misconfiguration: S3_BUCKET_NAME is not set.", 500)

    user_id = validation["user_id"]
    deployment_id = validation["deployment_id"]
    prompt_text = validation["prompt"]
    quality = validation["quality"]
    size = validation["size"]
    api_type_by_user = validation["api_type"]

    try:
        initialize_client(api_type_by_user, request.args.get("AI-KEY"))
    except Exception as e:
        return error_response(f"Failed to initialize provider client: {str(e)}", 500)

    number_of_tokens = num_tokens_from_string(str(prompt_text))
    object_key = build_object_key(api_type_by_user, user_id, deployment_id, prompt_text)

    if api_type_by_user == "azure":
        print(f"Generating file {object_key} using Azure OpenAI service")
        model_name = "Dalle3"
        image_url, status = generate_and_upload_obj_to_s3(model_name, prompt_text, size, quality, object_key)
    elif api_type_by_user == "open ai":
        print(f"Generating file {object_key} using OpenAI service")
        model_name = "dall-e-3"
        image_url, status = generate_and_upload_obj_to_s3(model_name, prompt_text, size, quality, object_key)
    else:
        print(f"Generating file {object_key} using Bedrock service")
        model_name = "amazon.titan-image-generator-v1"
        seed = 0
        image_url, status = generate_using_bedrock_and_upload_obj_to_s3(model_name, prompt_text, size, seed, quality, object_key)

    if image_url is None:
        return error_response("Image generation failed: no URL returned.", 502)
    if status == "ERROR":
        return error_response("Image generation failed.", 502, [str(image_url)])

    return jsonify({"URL": image_url, "token": number_of_tokens}), 200
            
#########################################################
 
@app.route("/remove", methods=["POST"])
def remove():
    print("remove")
    if not request.form.get("user_id") or not request.form.get("deployment_id"):
        return error_response("Missing form fields: user_id, deployment_id", 400)

    try:
        remove_pdfs(request.form["user_id"], request.form["deployment_id"])
        return jsonify({"success": "Remove successful"}), 200
    except NameError:
        return error_response("Remove endpoint is not configured: Pinecone client/index missing.", 501)
    except Exception as e:
        return error_response(f"Remove failed: {str(e)}", 500)

def lambda_handler(event,context):
    print(event)

    if awsgi is None:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "awsgi is not installed in this environment."}),
        }

    query_params = event.get("queryStringParameters") or {}
    api_type_user = query_params.get("API-TYPE")
    if not api_type_user:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "API-TYPE not present in request query string"}),
        }

    api_type_by_user = get_api_type(api_type_user.lower())
    if api_type_by_user not in ["azure", "open ai", "bedrock"]:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Invalid API-TYPE. Use one of: Azure Open AI, Open AI, Bedrock."}),
        }

    try:
        initialize_client(api_type_by_user, query_params.get("AI-KEY"))
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Failed to initialize provider client: {str(e)}"}),
        }

    event['httpMethod'] = event['requestContext']['http']['method']
    event['path'] = event['requestContext']['http']['path'] 
    
    print(event)

    ResponseToApi = awsgi.response(app, event, context)
    print(ResponseToApi)
    return ResponseToApi 


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
