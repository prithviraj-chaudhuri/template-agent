"""Manual test script for MCP endpoints.

Run this with the agent server running to verify endpoints work correctly.

Usage:
    python tests/manual/test_mcp_endpoints.py
"""

import asyncio
import json
from typing import Any

import httpx


async def test_mcp_endpoints() -> None:
    """Test MCP endpoint against a running agent server."""
    base_url = "http://localhost:8123"  # Default Aegra port

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test: List all servers with live status
        print("=" * 60)
        print("TEST: GET /mcp/servers (Consolidated Endpoint)")
        print("=" * 60)
        try:
            response = await client.get(f"{base_url}/mcp/servers")
            print(f"Status: {response.status_code}")

            if response.status_code != 200:
                print(f"✗ Expected 200, got {response.status_code}")
                print(f"Response: {response.text}")
                return

            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")

            servers = data.get("servers", [])
            print(f"\n{'=' * 60}")
            print(f"SUMMARY: Found {len(servers)} server(s)")
            print(f"{'=' * 60}")

            for server in servers:
                name = server.get("name", "unknown")
                status = server.get("status", "unknown")
                tools = server.get("tools", 0)
                prompts = server.get("prompts", 0)
                resources = server.get("resources", 0)
                error = server.get("error")

                print(f"\nServer: {name}")
                print(f"  URL: {server.get('url', 'N/A')}")
                print(f"  Status: {status}")

                if status == "connected":
                    print(f"  ✓ CONNECTED")
                    print(f"    - Tools: {tools}")
                    print(f"    - Prompts: {prompts}")
                    print(f"    - Resources: {resources}")
                    total = tools + prompts + resources
                    if total > 0:
                        print(f"  ✓ Total capabilities: {total}")
                    else:
                        print(f"  ⚠ WARNING: No capabilities found (tools, prompts, resources all 0)")
                elif status == "needs-auth":
                    print(f"  ⚠ NEEDS AUTHENTICATION")
                    if error:
                        print(f"    Error: {error}")
                elif status == "disconnected":
                    print(f"  ✗ DISCONNECTED")
                    if error:
                        print(f"    Error: {error}")
                else:
                    print(f"  ? UNKNOWN STATUS: {status}")

            # Overall health check
            print(f"\n{'=' * 60}")
            print("HEALTH CHECK")
            print(f"{'=' * 60}")
            connected = sum(1 for s in servers if s.get("status") == "connected")
            needs_auth = sum(1 for s in servers if s.get("status") == "needs-auth")
            disconnected = sum(1 for s in servers if s.get("status") == "disconnected")

            print(f"  Connected: {connected}")
            print(f"  Needs Auth: {needs_auth}")
            print(f"  Disconnected: {disconnected}")

            if connected == len(servers):
                print(f"\n  ✓ All servers are connected")
            elif connected > 0:
                print(f"\n  ⚠ Some servers have issues")
            else:
                print(f"\n  ✗ No servers are connected")

        except Exception as exc:
            print(f"✗ Error: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("Testing MCP Endpoints")
    print("Ensure the agent server is running on http://localhost:8123\n")
    asyncio.run(test_mcp_endpoints())
