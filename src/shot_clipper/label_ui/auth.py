"""Caller-identity helpers, shared by the legacy app.py routes and the new
v2 API blueprints (label_ui/api/). Split out from app.py specifically so
api/* can import these without a circular import: app.py registers the v2
blueprints near its own top, before current_user() used to be defined
further down in the same file.
"""
import os
import re

from flask import abort, request

# Hosted multi-tenant mode: unset (the default) keeps every behavior below
# exactly as it's always been for the native/Docker single-user tool - one
# shared clips dir, one shared labels.json, no login. Set to "1" only in the
# GCP deployment, where the GCS-mounted DATA_ROOT holds every user's data
# side by side (DATA_ROOT/<user>/...) and Identity-Aware Proxy sits in front
# of Cloud Run verifying who's asking, so every path derived from a request
# has to be pinned under that caller's own subtree - nothing else stops one
# user's request from simply naming another user's files.
MULTI_USER = os.environ.get("SHOT_CLIPPER_MULTI_USER") == "1"
# who can see /admin (and, in v2, manage the admin-managed label-category
# list) - the owner only, by default. Override with a comma-separated list
# if that ever needs to grow.
ADMIN_EMAILS = {e.strip() for e in
                os.environ.get("SHOT_CLIPPER_ADMIN_EMAILS", "tanpeng8847@gmail.com").split(",")
                if e.strip()}


def current_user() -> str | None:
    """Verified caller identity for this request, or None outside multi-user
    mode (there's only one user there: whoever's running the tool locally).
    Identity-Aware Proxy injects this header after verifying the caller
    against the allowlist configured on the Cloud Run service - IAP strips
    any such header an external caller tried to forge, so its presence here
    is trustworthy as long as ingress is actually locked to IAP-authenticated
    traffic (the deployment's job to guarantee, not this function's)."""
    if not MULTI_USER:
        return None
    email = request.headers.get("X-Goog-Authenticated-User-Email", "")
    email = email.removeprefix("accounts.google.com:")
    if not email:
        abort(401, "no verified identity - this deployment requires Identity-Aware Proxy")
    return email


def user_slug(email: str) -> str:
    """Filesystem-safe folder name for a user's data root."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", email)


# Outside multi-user mode there's no IAP identity at all (see current_user()
# above) - the rest of the app already treats that as "one shared local
# user, no login needed". The v2 API needs a concrete email to attribute
# every row to, so it extends that same convention with one fixed
# placeholder identity rather than requiring IAP for local/Docker dev use.
LOCAL_DEV_EMAIL = os.environ.get("SHOT_CLIPPER_LOCAL_USER_EMAIL", "local@localhost")


def require_user_email() -> str:
    """The email every v2 API route attributes its writes to."""
    return current_user() or LOCAL_DEV_EMAIL


def is_admin(email: str) -> bool:
    """Non-aborting admin check, for v2 moderation logic (delete-any-label/
    comment) that needs to ask "is this caller an admin?" without also
    encoding require_admin()'s "and 404 outside multi-user mode" behavior -
    the v2 API works in both modes (see require_user_email())."""
    return email in ADMIN_EMAILS


def require_admin() -> str:
    """Abort unless the caller is on the admin allowlist. Multi-user-only -
    the native/local tool has no concept of "other users' usage" to show."""
    if not MULTI_USER:
        abort(404)
    user = current_user()
    if user not in ADMIN_EMAILS:
        abort(403, "not authorized to view this page")
    return user
