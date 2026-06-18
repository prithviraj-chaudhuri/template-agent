"""Unit tests for MCP endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from deep_agent.aegra.feedback import app

    return TestClient(app)


@pytest.fixture
def mock_agent_config():
    """Mock agent_config.get_mcp_servers()."""
    with patch("deep_agent.aegra.mcp_endpoints.agent_config") as mock:
        yield mock


@pytest.fixture
def mock_get_server_capabilities():
    """Mock _get_server_capabilities() function."""
    with patch("deep_agent.aegra.mcp_endpoints._get_server_capabilities") as mock:
        yield mock


class TestGetMCPServers:
    """Tests for GET /mcp/servers endpoint (consolidated)."""

    @pytest.mark.asyncio
    async def test_list_servers_with_live_status_success(
        self, test_client, mock_agent_config, mock_get_server_capabilities
    ):
        """Should return list of enabled servers with live status and capabilities."""
        mock_agent_config.get_mcp_servers.return_value = {
            "server1": {
                "url": "http://localhost:5001/mcp",
                "transport": "streamable_http",
                "enabled": True,
                "auth": True,
                "ssl_verify": False,
                "timeout": 30,
            },
            "server2": {
                "url": "http://localhost:5002/mcp",
                "transport": "sse",
                "enabled": True,
                "timeout": 20,
            },
            "disabled-server": {
                "url": "http://localhost:5003/mcp",
                "enabled": False,  # Should be excluded
            },
        }

        # Mock successful responses for both enabled servers
        async def mock_capabilities(name, config):
            if name == "server1":
                return {
                    "name": "server1",
                    "url": "http://localhost:5001/mcp",
                    "status": "connected",
                    "tools": 3,
                    "prompts": 1,
                    "resources": 2,
                    "error": None,
                }
            else:
                return {
                    "name": "server2",
                    "url": "http://localhost:5002/mcp",
                    "status": "connected",
                    "tools": 2,
                    "prompts": 0,
                    "resources": 1,
                    "error": None,
                }

        mock_get_server_capabilities.side_effect = mock_capabilities

        response = test_client.get("/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert "servers" in data

        # Should only include enabled servers
        assert len(data["servers"]) == 2

        # Check that both enabled servers are present
        server_names = [s["name"] for s in data["servers"]]
        assert "server1" in server_names
        assert "server2" in server_names
        assert "disabled-server" not in server_names

    @pytest.mark.asyncio
    async def test_server_connected_with_capabilities(
        self, test_client, mock_agent_config, mock_get_server_capabilities
    ):
        """Should return connected status with tool/prompt/resource counts."""
        mock_agent_config.get_mcp_servers.return_value = {
            "test-server": {
                "url": "http://localhost:5001/mcp",
                "enabled": True,
            }
        }

        # Mock successful capabilities fetch
        async def mock_capabilities(name, config):
            return {
                "name": "test-server",
                "url": "http://localhost:5001/mcp",
                "status": "connected",
                "tools": 4,
                "prompts": 2,
                "resources": 1,
                "error": None,
                "auth_url": None,
            }

        mock_get_server_capabilities.side_effect = mock_capabilities

        response = test_client.get("/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert len(data["servers"]) == 1

        server = data["servers"][0]
        assert server["name"] == "test-server"
        assert server["url"] == "http://localhost:5001/mcp"
        assert server["status"] == "connected"
        assert server["tools"] == 4
        assert server["prompts"] == 2
        assert server["resources"] == 1
        assert server["error"] is None
        assert server["auth_url"] is None

    @pytest.mark.asyncio
    async def test_server_needs_auth(
        self, test_client, mock_agent_config, mock_get_server_capabilities
    ):
        """Should detect authentication errors and return needs-auth status."""
        mock_agent_config.get_mcp_servers.return_value = {
            "auth-server": {"url": "http://localhost:5001/mcp", "enabled": True}
        }

        # Mock auth error response
        async def mock_capabilities(name, config):
            return {
                "name": "auth-server",
                "url": "http://localhost:5001/mcp",
                "status": "needs-auth",
                "tools": 0,
                "prompts": 0,
                "resources": 0,
                "error": "Authentication required",
                "auth_url": None,
            }

        mock_get_server_capabilities.side_effect = mock_capabilities

        response = test_client.get("/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert len(data["servers"]) == 1

        server = data["servers"][0]
        assert server["name"] == "auth-server"
        assert server["status"] == "needs-auth"
        assert server["tools"] == 0
        assert server["prompts"] == 0
        assert server["resources"] == 0
        assert server["error"] == "Authentication required"
        assert "auth_url" in server

    @pytest.mark.asyncio
    async def test_server_needs_auth_with_oauth_url(
        self, test_client, mock_agent_config, mock_get_server_capabilities
    ):
        """Should include OAuth authorization URL when server supports OAuth."""
        mock_agent_config.get_mcp_servers.return_value = {
            "oauth-server": {"url": "https://api.example.com/mcp", "enabled": True}
        }

        # Mock auth error response with OAuth URL
        async def mock_capabilities(name, config):
            return {
                "name": "oauth-server",
                "url": "https://api.example.com/mcp",
                "status": "needs-auth",
                "tools": 0,
                "prompts": 0,
                "resources": 0,
                "error": "Authentication required",
                "auth_url": "https://auth.example.com/authorize?response_type=code&client_id=aegra-agent&redirect_uri=http%3A%2F%2Flocalhost%3A8123%2Foauth%2Fcallback&scope=mcp.tools",
            }

        mock_get_server_capabilities.side_effect = mock_capabilities

        response = test_client.get("/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert len(data["servers"]) == 1

        server = data["servers"][0]
        assert server["name"] == "oauth-server"
        assert server["status"] == "needs-auth"
        assert server["auth_url"] is not None
        assert "https://auth.example.com/authorize" in server["auth_url"]
        assert "response_type=code" in server["auth_url"]
        assert "client_id=aegra-agent" in server["auth_url"]

    @pytest.mark.asyncio
    async def test_server_disconnected(
        self, test_client, mock_agent_config, mock_get_server_capabilities
    ):
        """Should handle connection errors and return disconnected status."""
        mock_agent_config.get_mcp_servers.return_value = {
            "offline-server": {"url": "http://localhost:5001/mcp", "enabled": True}
        }

        # Mock connection error response
        async def mock_capabilities(name, config):
            return {
                "name": "offline-server",
                "url": "http://localhost:5001/mcp",
                "status": "disconnected",
                "tools": 0,
                "prompts": 0,
                "resources": 0,
                "error": "Connection refused",
            }

        mock_get_server_capabilities.side_effect = mock_capabilities

        response = test_client.get("/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert len(data["servers"]) == 1

        server = data["servers"][0]
        assert server["name"] == "offline-server"
        assert server["status"] == "disconnected"
        assert server["tools"] == 0
        assert server["prompts"] == 0
        assert server["resources"] == 0
        assert "Connection refused" in server["error"]

    @pytest.mark.asyncio
    async def test_mixed_server_statuses(
        self, test_client, mock_agent_config, mock_get_server_capabilities
    ):
        """Should handle multiple servers with different statuses."""
        mock_agent_config.get_mcp_servers.return_value = {
            "connected-server": {"url": "http://localhost:5001/mcp", "enabled": True},
            "auth-server": {"url": "http://localhost:5002/mcp", "enabled": True},
            "offline-server": {"url": "http://localhost:5003/mcp", "enabled": True},
        }

        # Mock different responses for each server
        async def mock_capabilities(name, config):
            responses = {
                "connected-server": {
                    "name": "connected-server",
                    "url": "http://localhost:5001/mcp",
                    "status": "connected",
                    "tools": 5,
                    "prompts": 1,
                    "resources": 2,
                    "error": None,
                },
                "auth-server": {
                    "name": "auth-server",
                    "url": "http://localhost:5002/mcp",
                    "status": "needs-auth",
                    "tools": 0,
                    "prompts": 0,
                    "resources": 0,
                    "error": "Authentication required",
                },
                "offline-server": {
                    "name": "offline-server",
                    "url": "http://localhost:5003/mcp",
                    "status": "disconnected",
                    "tools": 0,
                    "prompts": 0,
                    "resources": 0,
                    "error": "Connection timeout",
                },
            }
            return responses[name]

        mock_get_server_capabilities.side_effect = mock_capabilities

        response = test_client.get("/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert len(data["servers"]) == 3

        # Check each server has the correct status
        servers_by_name = {s["name"]: s for s in data["servers"]}

        assert servers_by_name["connected-server"]["status"] == "connected"
        assert servers_by_name["connected-server"]["tools"] == 5

        assert servers_by_name["auth-server"]["status"] == "needs-auth"
        assert servers_by_name["auth-server"]["error"] == "Authentication required"

        assert servers_by_name["offline-server"]["status"] == "disconnected"
        assert "timeout" in servers_by_name["offline-server"]["error"].lower()

    def test_list_servers_empty(self, test_client, mock_agent_config):
        """Should return empty list when no servers are enabled."""
        mock_agent_config.get_mcp_servers.return_value = {
            "disabled1": {"enabled": False},
            "disabled2": {"enabled": False},
        }

        response = test_client.get("/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert data["servers"] == []

    def test_list_servers_none_configured(self, test_client, mock_agent_config):
        """Should return empty list when no servers configured at all."""
        mock_agent_config.get_mcp_servers.return_value = {}

        response = test_client.get("/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert data["servers"] == []

    def test_list_servers_config_error(self, test_client, mock_agent_config):
        """Should return 500 when config loading fails."""
        mock_agent_config.get_mcp_servers.side_effect = Exception(
            "Config file not found"
        )

        response = test_client.get("/mcp/servers")

        assert response.status_code == 500
        assert "Failed to load MCP server configuration" in response.json()["detail"]
