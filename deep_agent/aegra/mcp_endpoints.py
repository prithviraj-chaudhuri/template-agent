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

from deep_agent.src.agent.config import agent_config
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/mcp", tags=["mcp"])


async def _extract_oauth_auth_url(
    server_url: str, exc: BaseException
) -> str | None:
    """Extract and build OAuth authorization URL from 401 response.

    Args:
        server_url: MCP server base URL
        exc: Exception that may contain WWW-Authenticate header

    Returns:
        Complete OAuth authorization URL if extractable, None otherwise
    """
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

    # Build authorization URL with code flow parameters
    from urllib.parse import urlencode

    # Get scopes from protected resource metadata
    scope = (
        " ".join(protected_resource_metadata.scopes_supported)
        if protected_resource_metadata.scopes_supported
        else None
    )

    # Build authorization URL
    auth_endpoint = str(auth_metadata.authorization_endpoint)
    params = {
        "response_type": "code",
        "client_id": "aegra-agent",  # This should match your OAuth client registration
        "redirect_uri": "http://localhost:8123/oauth/callback",  # Redirect endpoint
    }
    if scope:
        params["scope"] = scope

    return f"{auth_endpoint}?{urlencode(params)}"


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
            auth_url = await _extract_oauth_auth_url(config.get("url", ""), exc)

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
