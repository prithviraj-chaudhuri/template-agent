"""MCP server management HTTP endpoints.

Provides REST API for:
- Listing configured MCP servers with live connectivity and capability counts

Endpoints:
    GET /mcp/servers: List all configured MCP servers with status and capabilities
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from deep_agent.src.agent.config import agent_config
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/mcp", tags=["mcp"])

# In-memory storage for OAuth sessions (keyed by server_name)
# In production, use Redis or a database with TTL
# Maps server_name → {"client_id": str, "code_verifier": str}
_OAUTH_SESSIONS: dict[str, dict[str, str]] = {}


def _get_redirect_uri() -> str:
    """Construct OAuth redirect URI from current agent settings.

    Returns:
        Complete redirect URI (e.g., http://localhost:5002/mcp/oauth/callback)
    """
    host = settings.AGENT_HOST
    port = settings.AGENT_PORT

    # Use localhost for 0.0.0.0 bind address
    if host == "0.0.0.0":
        host = "localhost"

    return f"http://{host}:{port}/mcp/oauth/callback"


async def _extract_oauth_auth_url(
    server_url: str, exc: BaseException, server_name: str
) -> str | None:
    """Extract and build OAuth authorization URL from 401 response.

    Args:
        server_url: MCP server base URL
        exc: Exception that may contain WWW-Authenticate header
        server_name: Name of the MCP server for session tracking

    Returns:
        Complete OAuth authorization URL if extractable, None otherwise
    """
    # Check if we already have an active session for this server
    if server_name in _OAUTH_SESSIONS:
        session = _OAUTH_SESSIONS[server_name]
        logger.info(
            "Reusing existing OAuth session for %s (client_id: %s)",
            server_name,
            session["client_id"]
        )

        # Rebuild the authorization URL with the existing client_id
        redirect_uri = _get_redirect_uri()

        # We need to regenerate PKCE since the old one might have been used
        import base64
        import hashlib
        import secrets

        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')

        # Update the code_verifier in the session
        session["code_verifier"] = code_verifier

        # Return auth URL with existing client_id
        from urllib.parse import urlencode
        auth_url = session.get("auth_endpoint", "https://auth.atlassian.com/authorize")
        params = {
            "response_type": "code",
            "client_id": session["client_id"],
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        scope = session.get("scope")
        if scope:
            params["scope"] = scope

        return f"{auth_url}?{urlencode(params)}"

    import httpx
    from mcp.client.auth.utils import (
        extract_field_from_www_auth,
        build_protected_resource_metadata_discovery_urls,
        build_oauth_authorization_server_metadata_discovery_urls,
        handle_protected_resource_response,
        handle_auth_metadata_response,
        create_oauth_metadata_request,
    )

    # Extract the 401 response from the exception chain
    response = None
    for sub in getattr(exc, "exceptions", [exc]):
        if hasattr(sub, "response"):
            response = sub.response
            break
        if hasattr(sub, "__cause__") and hasattr(sub.__cause__, "response"):
            response = sub.__cause__.response
            break

    if not response or not hasattr(response, "status_code"):
        return None

    if response.status_code != 401:
        return None

    # Extract resource_metadata URL from WWW-Authenticate header
    www_auth_url = extract_field_from_www_auth(response, "resource_metadata")

    # Build discovery URLs per SEP-985
    prm_urls = build_protected_resource_metadata_discovery_urls(
        www_auth_url, server_url
    )

    # Try to fetch protected resource metadata
    protected_resource_metadata = None
    async with httpx.AsyncClient(verify=False) as client:  # nosec B501
        for prm_url in prm_urls:
            try:
                prm_request = create_oauth_metadata_request(prm_url)
                prm_response = await client.send(prm_request)
                protected_resource_metadata = await handle_protected_resource_response(
                    prm_response
                )
                if protected_resource_metadata:
                    break
            except Exception:
                continue

    if not protected_resource_metadata:
        return None

    # Get authorization server metadata URL
    auth_server_url = (
        str(protected_resource_metadata.authorization_servers[0])
        if protected_resource_metadata.authorization_servers
        else None
    )

    # Build OAuth authorization server metadata discovery URLs
    asm_urls = build_oauth_authorization_server_metadata_discovery_urls(
        auth_server_url, server_url
    )

    # Try to fetch authorization server metadata
    auth_metadata = None
    async with httpx.AsyncClient(verify=False) as client:  # nosec B501
        for asm_url in asm_urls:
            try:
                asm_request = create_oauth_metadata_request(asm_url)
                asm_response = await client.send(asm_request)
                should_continue, auth_metadata = await handle_auth_metadata_response(
                    asm_response
                )
                if auth_metadata:
                    break
                if not should_continue:
                    break
            except Exception:
                continue

    if not auth_metadata or not auth_metadata.authorization_endpoint:
        return None

    # Perform dynamic client registration first
    from urllib.parse import urlencode
    import httpx
    import secrets
    import hashlib
    import base64

    registration_endpoint = str(auth_metadata.registration_endpoint) if auth_metadata.registration_endpoint else None

    if not registration_endpoint:
        logger.warning("No registration endpoint found in OAuth metadata - cannot use dynamic client registration")
        return None

    # Get scopes from protected resource metadata
    scope = (
        " ".join(protected_resource_metadata.scopes_supported)
        if protected_resource_metadata.scopes_supported
        else None
    )

    # Register client dynamically (RFC 7591)
    redirect_uri = _get_redirect_uri()
    base_uri = f"http://{settings.AGENT_HOST if settings.AGENT_HOST != '0.0.0.0' else 'localhost'}:{settings.AGENT_PORT}"

    try:
        async with httpx.AsyncClient(verify=False) as client:  # nosec B501
            registration_response = await client.post(
                registration_endpoint,
                json={
                    "client_name": "Aegra Agent",
                    "client_uri": base_uri,
                    "redirect_uris": [redirect_uri],
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "token_endpoint_auth_method": "none",  # Public client (no secret)
                    "scope": scope,
                },
                headers={"Content-Type": "application/json"},
            )

            if registration_response.status_code not in (200, 201):
                logger.error(
                    "Dynamic client registration failed: %s - %s",
                    registration_response.status_code,
                    registration_response.text
                )
                return None

            registration_data = registration_response.json()
            client_id = registration_data.get("client_id")

            if not client_id:
                logger.error("No client_id in registration response")
                return None

            logger.info("Successfully registered OAuth client: %s for server: %s", client_id, server_name)

            # Generate PKCE code_verifier and code_challenge (RFC 7636)
            code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode('utf-8')).digest()
            ).decode('utf-8').rstrip('=')

            # Store session data keyed by server_name (not state - UI overwrites state)
            # In production, use Redis or a database with TTL
            _OAUTH_SESSIONS[server_name] = {
                "client_id": client_id,
                "code_verifier": code_verifier,
                "auth_endpoint": str(auth_metadata.authorization_endpoint),
                "scope": scope,
            }

            # Build authorization URL with dynamically registered client_id and PKCE
            # Don't include state - the UI will add its own for CSRF protection
            auth_endpoint = str(auth_metadata.authorization_endpoint)
            params = {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
            if scope:
                params["scope"] = scope

            return f"{auth_endpoint}?{urlencode(params)}"

    except Exception as e:
        logger.error("Failed to perform dynamic client registration: %s", e, exc_info=True)
        return None


async def _get_server_capabilities(server_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Fetch tools, prompts, and resources count from a single MCP server.

    Args:
        server_name: Name of the MCP server to check.
        config: Server configuration from mcp.json.

    Returns:
        {
            "name": str,
            "url": str,
            "status": "connected" | "needs-auth" | "disconnected",
            "tools": int,
            "prompts": int,
            "resources": int,
            "error": str | None,
            "auth_url": str | None  # OAuth authorization URL when needs-auth
        }
    """
    from deep_agent.aegra.mcp import _build_server_config, _is_auth_error

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        timeout = config.get("timeout", 30)

        # Build client config without SSO token for discovery
        client_config = _build_server_config(config, sso_token=None)

        tools_count = 0
        prompts_count = 0
        resources_count = 0

        async with asyncio.timeout(timeout):
            client = MultiServerMCPClient({server_name: client_config})

            # Use the session context to access the underlying MCP protocol
            async with client.session(server_name) as session:
                # Fetch lists using the MCP protocol methods
                tools_result, prompts_result, resources_result = await asyncio.gather(
                    session.list_tools(),
                    session.list_prompts(),
                    session.list_resources(),
                    return_exceptions=True,
                )

                # Count successful results from MCP protocol responses
                if not isinstance(tools_result, Exception) and hasattr(
                    tools_result, "tools"
                ):
                    tools_count = len(tools_result.tools)
                if not isinstance(prompts_result, Exception) and hasattr(
                    prompts_result, "prompts"
                ):
                    prompts_count = len(prompts_result.prompts)
                if not isinstance(resources_result, Exception) and hasattr(
                    resources_result, "resources"
                ):
                    resources_count = len(resources_result.resources)

        logger.info(
            "MCP server '%s' connected: %d tool(s), %d prompt(s), %d resource(s)",
            server_name,
            tools_count,
            prompts_count,
            resources_count,
        )

        return {
            "name": server_name,
            "url": config.get("url", ""),
            "status": "connected",
            "tools": tools_count,
            "prompts": prompts_count,
            "resources": resources_count,
            "error": None,
            "auth_url": None,
        }

    except Exception as exc:
        error_msg = str(exc)
        logger.warning(
            "MCP server '%s' connection failed: %s",
            server_name,
            error_msg,
            exc_info=True,
        )

        # Detect authentication errors (401/403)
        if _is_auth_error(exc):
            # Try to extract OAuth authorization URL
            auth_url = await _extract_oauth_auth_url(config.get("url", ""), exc, server_name)

            return {
                "name": server_name,
                "url": config.get("url", ""),
                "status": "needs-auth",
                "tools": 0,
                "prompts": 0,
                "resources": 0,
                "error": "Authentication required",
                "auth_url": auth_url,
            }

        # All other errors are connection failures
        return {
            "name": server_name,
            "url": config.get("url", ""),
            "status": "disconnected",
            "tools": 0,
            "prompts": 0,
            "resources": 0,
            "error": error_msg[:200],  # Truncate long error messages
            "auth_url": None,
        }


@router.get("/oauth/callback", response_class=HTMLResponse)
async def oauth_callback(code: str, state: str | None = None) -> str:
    """Handle OAuth callback after user authorization.

    This endpoint receives the authorization code from the OAuth provider
    and exchanges it for an access token using PKCE.

    Args:
        code: Authorization code from OAuth provider
        state: Optional state parameter (used by UI for CSRF, not session tracking)

    Returns:
        HTML success page with auto-close script

    Raises:
        HTTPException: 400 if code exchange fails
    """
    import httpx

    # Since we can't rely on state (UI overwrites it), we need to find the session
    # For now, assume there's only one OAuth server in progress
    if not _OAUTH_SESSIONS:
        logger.error("No active OAuth sessions found")
        raise HTTPException(
            status_code=400,
            detail="No active OAuth session found. Please restart the authentication flow."
        )

    # Get the first (and likely only) OAuth session
    # In a multi-tenant setup, you'd need a better way to track this
    server_name = next(iter(_OAUTH_SESSIONS.keys()))
    session = _OAUTH_SESSIONS[server_name]
    client_id = session["client_id"]
    code_verifier = session["code_verifier"]

    logger.info("Processing OAuth callback for server: %s, client_id: %s", server_name, client_id)

    # Get server config to find token endpoint
    try:
        servers_config = agent_config.get_mcp_servers()
        if server_name not in servers_config:
            raise HTTPException(status_code=400, detail="Server not found")

        server_config = servers_config[server_name]
        server_url = server_config.get("url", "")

        # Discover token endpoint from OAuth metadata
        # For Atlassian, it's https://auth.atlassian.com/oauth/token
        token_endpoint = "https://auth.atlassian.com/oauth/token"

        # Exchange authorization code for access token with PKCE
        redirect_uri = _get_redirect_uri()

        async with httpx.AsyncClient(verify=False) as client:  # nosec B501
            token_response = await client.post(
                token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": client_id,
                    "redirect_uri": redirect_uri,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if token_response.status_code != 200:
                logger.error(
                    "Token exchange failed: %s - %s",
                    token_response.status_code,
                    token_response.text
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Token exchange failed: {token_response.text}"
                )

            token_data = token_response.json()
            access_token = token_data.get("access_token")
            refresh_token = token_data.get("refresh_token")

            logger.info(
                "Successfully obtained access token for server: %s",
                server_name
            )

            # Clean up session data
            del _OAUTH_SESSIONS[server_name]

            # TODO: Store access_token and refresh_token for future use
            # For now, just return success HTML page
            # In production, store these in Redis or database keyed by server_name

            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authentication Successful</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    }
                    .container {
                        text-align: center;
                        background: white;
                        padding: 3rem;
                        border-radius: 1rem;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                        max-width: 400px;
                    }
                    h1 {
                        color: #10b981;
                        margin-bottom: 1rem;
                        font-size: 2rem;
                    }
                    p {
                        color: #6b7280;
                        margin-bottom: 1.5rem;
                        font-size: 1.1rem;
                    }
                    .checkmark {
                        font-size: 4rem;
                        color: #10b981;
                        margin-bottom: 1rem;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="checkmark">✓</div>
                    <h1>Authentication Successful!</h1>
                    <p>You can close this window and return to the application.</p>
                    <p style="font-size: 0.9rem; color: #9ca3af;">Server: """ + (server_name or "unknown") + """</p>
                </div>
                <script>
                    // Notify parent window
                    if (window.opener) {
                        window.opener.postMessage({
                            type: 'oauth-success',
                            serverName: '""" + (server_name or "unknown") + """'
                        }, window.location.origin);
                    }
                    // Auto-close after 3 seconds
                    setTimeout(() => window.close(), 3000);
                </script>
            </body>
            </html>
            """

    except Exception as e:
        logger.error("OAuth callback failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"OAuth callback failed: {str(e)}"
        ) from e


@router.post("/oauth/reset")
async def reset_oauth_cache() -> dict[str, Any]:
    """Clear OAuth client cache to force re-registration.

    Use this when the redirect URI or other OAuth settings have changed.

    Returns:
        {"status": "success", "message": "OAuth cache cleared"}
    """
    global _OAUTH_SESSIONS
    _OAUTH_SESSIONS.clear()

    logger.info("OAuth cache cleared - clients will re-register on next connection attempt")

    return {
        "status": "success",
        "message": "OAuth cache cleared. Refresh MCP servers to re-register.",
        "current_redirect_uri": _get_redirect_uri()
    }


@router.get("/servers")
async def get_mcp_servers() -> dict[str, Any]:
    """Return all configured MCP servers with live status and capabilities.

    Connects to each enabled MCP server in parallel and returns their
    real-time status including tool, prompt, and resource counts.

    For servers requiring OAuth authentication (status: "needs-auth"),
    the response includes an auth_url field with the OAuth authorization
    endpoint that the user should visit to complete the code flow.

    Returns:
        {
            "servers": [
                {
                    "name": "template-mcp-server",
                    "url": "http://localhost:5001/mcp",
                    "status": "connected" | "needs-auth" | "disconnected",
                    "tools": 4,
                    "prompts": 0,
                    "resources": 0,
                    "error": null | "error message",
                    "auth_url": null | "https://auth.example.com/authorize?..."
                }
            ]
        }

    Raises:
        HTTPException: 500 if server configuration cannot be loaded.
    """
    try:
        servers_config = agent_config.get_mcp_servers()
    except Exception as exc:
        logger.error("Failed to load MCP server config", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load MCP server configuration: {exc}",
        ) from exc

    # Only check enabled servers
    enabled_servers = {
        name: config
        for name, config in servers_config.items()
        if config.get("enabled", True)
    }

    if not enabled_servers:
        logger.info("No enabled MCP servers found")
        return {"servers": []}

    # Fetch capabilities from all servers in parallel
    servers_list = await asyncio.gather(
        *[
            _get_server_capabilities(name, config)
            for name, config in enabled_servers.items()
        ]
    )

    logger.info("Returned %d MCP server(s) with status", len(servers_list))
    return {"servers": servers_list}
