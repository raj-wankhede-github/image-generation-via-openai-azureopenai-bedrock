import app as app_module


def _form_payload():
    return {
        "user_id": "u1",
        "deployment_id": "d1",
        "prompt": "a cat playing piano",
    }


def _query_openai():
    return "?API-TYPE=Open%20AI&Quality=standard&Size=1024x1024"


def test_prompt_missing_form_fields_returns_400():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post("/prompt" + _query_openai(), data={})

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Request validation failed."


def test_prompt_success_openai_path_returns_200(monkeypatch):
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    monkeypatch.setattr(app_module, "bucket_name", "test-bucket")
    monkeypatch.setattr(app_module, "initialize_client", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app_module,
        "generate_and_upload_obj_to_s3",
        lambda *_args, **_kwargs: ("https://test-bucket.s3.amazonaws.com/generated_images/test.jpg", "SUCCESS"),
    )

    response = client.post("/prompt" + _query_openai(), data=_form_payload())

    assert response.status_code == 200
    body = response.get_json()
    assert "URL" in body
    assert "token" in body


def test_prompt_provider_error_returns_502(monkeypatch):
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    monkeypatch.setattr(app_module, "bucket_name", "test-bucket")
    monkeypatch.setattr(app_module, "initialize_client", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app_module,
        "generate_and_upload_obj_to_s3",
        lambda *_args, **_kwargs: ("provider failed", "ERROR"),
    )

    response = client.post("/prompt" + _query_openai(), data=_form_payload())

    assert response.status_code == 502
    body = response.get_json()
    assert body["error"] == "Image generation failed."


def test_remove_missing_fields_returns_400():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post("/remove", data={})

    assert response.status_code == 400
    body = response.get_json()
    assert "Missing form fields" in body["error"]


def test_remove_unconfigured_returns_501(monkeypatch):
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    def _raise_name_error(*_args, **_kwargs):
        raise NameError("pinecone_index is not defined")

    monkeypatch.setattr(app_module, "remove_pdfs", _raise_name_error)

    response = client.post("/remove", data={"user_id": "u1", "deployment_id": "d1"})

    assert response.status_code == 501
    body = response.get_json()
    assert "not configured" in body["error"].lower()
