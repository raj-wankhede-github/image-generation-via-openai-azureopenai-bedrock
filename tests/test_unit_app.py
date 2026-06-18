import app as app_module


def test_slugify_value_basic():
    assert app_module.slugify_value("Hello, World!") == "hello_world"


def test_slugify_value_fallback_for_empty():
    assert app_module.slugify_value("   !!!   ") == "na"


def test_build_object_key_is_sanitized_and_structured(monkeypatch):
    monkeypatch.setattr(app_module.time, "time", lambda: 1700000000)
    object_key = app_module.build_object_key(
        "Open AI",
        "User@123",
        "Deployment#A",
        "A scenic mountain with birds!!!",
    )

    assert object_key.startswith("generated_images/open_ai/")
    assert object_key.endswith(".jpg")
    assert " " not in object_key
    assert "@" not in object_key
    assert "#" not in object_key


def test_validate_prompt_payload_missing_fields():
    result = app_module.validate_prompt_payload({}, {})
    assert result["ok"] is False
    assert any("Missing form fields" in msg for msg in result["errors"])
    assert any("Missing query parameters" in msg for msg in result["errors"])


def test_validate_prompt_payload_invalid_api_type():
    form = {"user_id": "u1", "deployment_id": "d1", "prompt": "cat"}
    query = {"API-TYPE": "Unknown", "Quality": "standard", "Size": "1024x1024"}
    result = app_module.validate_prompt_payload(form, query)
    assert result["ok"] is False
    assert "Invalid API-TYPE" in result["errors"][0]


def test_validate_prompt_payload_invalid_openai_quality():
    form = {"user_id": "u1", "deployment_id": "d1", "prompt": "cat"}
    query = {"API-TYPE": "Open AI", "Quality": "premium", "Size": "1024x1024"}
    result = app_module.validate_prompt_payload(form, query)
    assert result["ok"] is False
    assert "Invalid Quality for OpenAI/AzureOpenAI" in result["errors"][0]


def test_validate_prompt_payload_success_openai():
    form = {"user_id": "u1", "deployment_id": "d1", "prompt": "cat image"}
    query = {"API-TYPE": "Open AI", "Quality": "standard", "Size": "1024x1024"}
    result = app_module.validate_prompt_payload(form, query)
    assert result["ok"] is True
    assert result["api_type"] == "open ai"
    assert result["quality"] == "standard"
    assert result["size"] == "1024x1024"
