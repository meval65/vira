import asyncio
import httpx
import sys

BASE_URL = "http://localhost:5000/api"

async def test_endpoint(client, endpoint, name):
    try:
        response = await client.get(f"{BASE_URL}{endpoint}")
        if response.status_code == 200:
            print(f"✓ {name}: OK")
            return True
        else:
            print(f"✗ {name}: Failed ({response.status_code}) - {response.text[:100]}")
            return False
    except Exception as e:
        print(f"✗ {name}: Error - {e}")
        return False

async def main():
    print("Testing Brain API Stability...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        results = [
            await test_endpoint(client, "/personas", "List Personas"),
            await test_endpoint(client, "/personas/active", "Active Persona"),
            await test_endpoint(client, "/triples?limit=1", "Knowledge Graph"),
            await test_endpoint(client, "/neural-events?limit=1", "Neural Events")
        ]
        
        if all(results):
            print("\nAll tests passed! System is stable.")
            sys.exit(0)
        else:
            print("\nSome tests failed.")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
