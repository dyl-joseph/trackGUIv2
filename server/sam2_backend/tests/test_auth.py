from server.sam2_backend.auth import bearer_token_is_valid


def test_bearer_auth_is_optional_only_without_a_configured_token():
    assert bearer_token_is_valid(None, None)
    assert bearer_token_is_valid("", "Bearer anything")
    assert not bearer_token_is_valid("secret", None)
    assert not bearer_token_is_valid("secret", "Basic secret")
    assert not bearer_token_is_valid("secret", "Bearer wrong")
    assert bearer_token_is_valid("secret", "Bearer secret")
