import secrets


def bearer_token_is_valid(configured_token, authorization):
    if not configured_token:
        return True
    scheme, _, supplied_token = (authorization or "").partition(" ")
    return scheme.lower() == "bearer" and secrets.compare_digest(
        supplied_token, configured_token
    )
