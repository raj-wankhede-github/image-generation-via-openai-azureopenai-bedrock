from flask import Flask, request
import requests
import os, shutil, json, sys, time, awsgi, tiktoken
from io import BytesIO
from openai import AzureOpenAI
from openai import OpenAI

import asyncio
import boto3
from botocore.exceptions import ClientError
import base64

# This initializes the clients Bedrock Runtime and S3
bedrock_runtime_client = boto3.client('bedrock-runtime', region_name='us-east-1')
s3 = boto3.client('s3')
bucket_name = os.getenv("S3_BUCKET_NAME")

app = Flask(__name__)
app.config["TIMEOUT"] = 60  # sets the timeout limit to 60 seconds

def generate_using_bedrock_and_upload_obj_to_s3(model_name, prompt, size, seed, Quality, object_key):
    try:
        ## user pass input as width x height
        width = int(size.split("x")[0])
        height = int(size.split("x")[1])

        request = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": prompt},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": Quality,
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
    user_id = request.form["user_id"]
    deployment_id = request.form["deployment_id"]
    prompt = request.form["prompt"]
    Quality_user = request.args.get("Quality").lower()
    Quality = request.args.get("Quality").lower()
    Size = request.args.get("Size")

    api_type = request.args.get('API-TYPE').lower()
    api_type_by_user = get_api_type(api_type)
    print(api_type_by_user)

    Size_values = ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
    
    # https://platform.openai.com/docs/guides/images/generations
    # When using DALL·E 3, images can have a size of 1024x1024, 1024x1792 or 1792x1024 pixels.
    Size_values = ['1024x1024', '1792x1024', '1024x1792']
    Quality_values = ['standard', 'hd']
    
    validity_check = ['azure', 'open ai']

    if api_type_by_user in validity_check:
        try:
            if Size in Size_values:
                print(f"Size value is present: {Size}")
                if Quality in Quality_values:
                    print(f"Quality value is present: {Quality}")
                else:
                    error_message = {
                        "error": f"Quality '{Quality_user}' NOT present for OpenAI/AzureOpenAI. Possible Quality values are: {Quality_values}"
                    }
                    return error_message            
            else:
                error_message = {
                    "error": f"Size '{Size}' NOT present for OpenAI/AzureOpenAI. Possible Size values are: {Size_values}"
                }
                return error_message
        except Exception as e:
            print(str(e))
            if "Error code: 400" in str(e):
                return {
                    "error": str(e),
                }
            else: 
                return {
                    "error": "There was an error processing request."
                }
    
    else:
        # Below values for bedrock amazon.titan-image-generator-v1 as of 8th April 2024
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
        Quality_values = ['standard', 'premium'] 
        Size_values = ['1024x1024', '768x768', '512x512', '768x1152', '384x576', '1152x768', '576x384', '768x1280', '384x640', '1280x768', '640x384', '896x1152', '448x576', '1152x896', '576x448', '768x1408', '384x704', '1408x768', '704x384', '640x1408', '320x704', '1408x640', '704x320', '1152x640', '1173x640']
        
        try:
            if Size in Size_values:
                print(f"Size value is present: {Size}")
                if Quality in Quality_values:
                    print(f"Quality value is present: {Quality}")
                else:
                    error_message = {
                        "error": f"Quality '{Quality_user}' NOT present for Bedrock. Possible Quality values are: {Quality_values}"
                    }
                    return error_message            
            else:
                error_message = {
                    "error": f"Size '{Size}' NOT present for Bedrock. Possible Size values are: {Size_values}"
                }
                return error_message
        except Exception as e:
            print(str(e))
            if "Error code: 400" in str(e):
                return {
                    "error": str(e),
                }
            else: 
                return {
                    "error": "There was an error processing request."
                }

#########################################################

    number_of_tokens = num_tokens_from_string(str(prompt))
    object_key = f"generated_images/{api_type.replace(' ', '_')}_{user_id}_{deployment_id}_{prompt.replace(' ', '_')}.jpg"

    if "azure" in api_type_by_user :
        print(f"Generating file {object_key} using Azure Openai service")
        model_name = "Dalle3"
        image_url, status = generate_and_upload_obj_to_s3(model_name, prompt, Size, Quality, object_key)

    elif "open ai" in api_type_by_user:
        print(f"Generating file {object_key} using Openai service")
        model_name = "dall-e-3"
        image_url, status = generate_and_upload_obj_to_s3(model_name, prompt, Size, Quality, object_key)
        
    elif "bedrock" in api_type_by_user:    
        print(f"Generating file {object_key} using Bedrock service")
        model_name = "amazon.titan-image-generator-v1"
        seed = 0
        image_url, status = generate_using_bedrock_and_upload_obj_to_s3(model_name, prompt, Size, seed, Quality, object_key)
            
    if image_url is None:
        return {
            "error": "No URL returned."
        }
    if "ERROR" in status:
        return {
            "error": image_url
        }

    return  {
        "URL": image_url,
        "token":number_of_tokens
    }
            
#########################################################
 
@app.route("/remove", methods=["POST"])
def remove():
    print("remove")
    response = remove_pdfs(request.form["user_id"], request.form["deployment_id"])
    return_json_message = {
        "success" : "Remove successful"
    }
    return return_json_message

def lambda_handler(event,context):
    print(event)

    if event['rawQueryString']:
         
        if 'API-TYPE' in event['queryStringParameters']:
            print(f"API Type provided by user: {event['queryStringParameters']['API-TYPE']}")
            api_type_user = event['queryStringParameters']['API-TYPE']
            api_type_by_user = api_type_user.lower()
            api_type_by_user = get_api_type(api_type_by_user)
            
            global Client
            
            if "azure" in api_type_by_user :
                
                if 'AI-KEY' in event['queryStringParameters']:
                    print("AI-KEY query string exists. Using the one provided from AI-KEY.")
                    Client = AzureOpenAI(
                      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                      api_key=event['queryStringParameters']['AI-KEY'],  
                      api_version="2024-02-01"
                    )
                else:
                    print("AI-KEY not provided in query string in the request")
                    Client = AzureOpenAI(
                      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                      api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                      api_version="2024-02-01"
                    )                
                
            elif  "open ai" or "bedrock" in api_type_by_user:
                
                if 'AI-KEY' in event['queryStringParameters']:
                    print("AI-KEY query string exists. Using the one provided from AI-KEY.")
                    Client = OpenAI(api_key = event['queryStringParameters']['AI-KEY'])
                else:
                    print("AI-KEY not provided in query string in the request")
                    Client = OpenAI()
            
        else:
            print("API-TYPE not present in request query string parameters")
            return_message = {"error" : "API-TYPE not present in request query string"}
            return return_message
            
    else:
        event['queryStringParameters'] = {} 
        return_message = {"error" : "API-TYPE not present in request query string"}
        return return_message

    event['httpMethod'] = event['requestContext']['http']['method']
    event['path'] = event['requestContext']['http']['path'] 
    
    print(event)

    ResponseToApi = awsgi.response(app, event, context)
    print(ResponseToApi)
    return ResponseToApi 
