# Image Generation via OpenAI, Azure OpenAI, and Amazon Bedrock

Flask + AWS Lambda (via `awsgi`) service that generates images using one of three providers:

- Azure OpenAI (`Dalle3` deployment)
- OpenAI (`dall-e-3`)
- Amazon Bedrock (`amazon.titan-image-generator-v1`)

Generated images are uploaded to Amazon S3 and returned as a public S3 URL.

## Features

- Unified `/prompt` endpoint for all three backends
- Input validation for `Size` and `Quality` by provider
- Optional per-request API key via query parameter (`AI-KEY`) for OpenAI/Azure OpenAI
- Token count for prompt text using `tiktoken`
- Sanitized/slugified S3 object key builder for generated images
- Clearer HTTP status codes for validation, provider, and configuration errors
- Lambda-compatible handler: `lambda_handler(event, context)`

## Tech Stack

- Python
- Flask
- AWS Lambda + `awsgi`
- OpenAI Python SDK (`OpenAI`, `AzureOpenAI`)
- Amazon Bedrock Runtime (boto3)
- Amazon S3 (boto3)

## Project Structure

- `app.py`: Main Flask app, provider routing logic, S3 upload, Lambda entrypoint
- `README.md`: Project documentation
- `requirements.txt`: Runtime and test dependencies
- `tests/`: Unit and integration test suite

## Prerequisites

- Python 3.10+
- AWS credentials configured (for Bedrock + S3 access)
- S3 bucket available
- For Azure OpenAI usage:
	- Azure OpenAI endpoint
	- Valid API key (env var or query parameter)
- For OpenAI usage:
	- OpenAI API key (env var or query parameter)

## Environment Variables

Required:

- `S3_BUCKET_NAME`: Target bucket where generated images are uploaded

For Azure OpenAI (if not passing `AI-KEY` in query string):

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`

For OpenAI (if not passing `AI-KEY` in query string):

- `OPENAI_API_KEY`

## Installation

1. Create and activate a virtual environment.
2. Install dependencies.

Example:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: `awsgi` is installed on non-Windows platforms via environment markers. This keeps local Windows test setup compatible while preserving Lambda deployment support.

## Running Locally

Run Flask app:

```bash
python app.py
```

`app.py` now includes a local entrypoint:

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

## Testing

Run tests:

```bash
pytest -q
```

Test coverage currently includes:

- Unit tests for payload validation and object key slugification
- Integration-style Flask route tests for `/prompt` and `/remove`

## API

### POST `/prompt`

Generate image and upload to S3.

#### Query Parameters

- `API-TYPE` (required): one of
	- `Azure Open AI`
	- `Open AI`
	- `Bedrock`
- `AI-KEY` (optional):
	- Used for Azure OpenAI/OpenAI when provided
	- Ignored for Bedrock
- `Quality` (required):
	- OpenAI/Azure OpenAI: `standard` or `hd`
	- Bedrock: `standard` or `premium`
- `Size` (required):
	- OpenAI/Azure OpenAI: `1024x1024`, `1792x1024`, `1024x1792`
	- Bedrock: see list in code (includes multiple portrait/landscape sizes)

#### Form Body

- `user_id` (required)
- `deployment_id` (required)
- `prompt` (required)

#### Response

Success:

```json
{
	"URL": "https://<bucket>.s3.amazonaws.com/generated_images/...jpg",
	"token": 12
}
```

Error:

```json
{
	"error": "<error message>"
}
```

#### Example cURL

OpenAI:

```bash
curl -X POST "http://localhost:5000/prompt?API-TYPE=Open%20AI&Quality=standard&Size=1024x1024" \
	-F "user_id=123" \
	-F "deployment_id=abc" \
	-F "prompt=A cinematic mountain landscape at sunrise"
```

Azure OpenAI (with per-request key):

```bash
curl -X POST "http://localhost:5000/prompt?API-TYPE=Azure%20Open%20AI&AI-KEY=<AZURE_KEY>&Quality=hd&Size=1024x1792" \
	-F "user_id=123" \
	-F "deployment_id=abc" \
	-F "prompt=Modern architectural building, ultra-detailed"
```

Bedrock:

```bash
curl -X POST "http://localhost:5000/prompt?API-TYPE=Bedrock&Quality=premium&Size=1024x1024" \
	-F "user_id=123" \
	-F "deployment_id=abc" \
	-F "prompt=A watercolor painting of a fox in a forest"
```

### POST `/remove`

Current code references Pinecone namespace deletion (`remove_pdfs`) but required Pinecone client/index setup is not present in this repository.

Status:

- Endpoint exists in code but is not production-ready without additional Pinecone wiring.

## AWS Lambda Usage

Entrypoint:

- `lambda_handler(event, context)`

Behavior:

- Reads `API-TYPE` and optional `AI-KEY` from query string
- Initializes provider client
- Maps API Gateway HTTP API event to Flask-compatible fields
- Delegates to Flask routes through `awsgi.response(...)`

## IAM Permissions

Ensure the runtime role has access to:

- `s3:PutObject` on your target bucket
- `bedrock:InvokeModel` for `amazon.titan-image-generator-v1` (if Bedrock path is used)

## Known Limitations

- `/remove` depends on undefined Pinecone objects (`pinecone_index`)
- No built-in auth/rate limiting around endpoints

## Future Improvements

- Add Pinecone client/index wiring for `/remove`
- Add request schema validation library (for example, `pydantic` or `marshmallow`)
- Add CI workflow to run tests automatically
- Add sanitized key hash suffix for collision resistance

## License

No license file is currently present in this repository.