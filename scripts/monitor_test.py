#!/usr/bin/env python3

import asyncio
import sys
from pathlib import Path

# Add the scripts directory to path so we can import
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from api_monitor import APIMonitor


async def test():
    monitor = APIMonitor()
    try:
        remote_spec = await monitor.fetch_remote_spec()
        print("✅ Successfully fetched remote spec")
        print(f"Version: {remote_spec.get('info', {}).get('version', 'unknown')}")
        print(f"Paths count: {len(remote_spec.get('paths', {}))}")

        local_spec = monitor.load_local_spec()
        print(
            f"Local spec version: {local_spec.get('info', {}).get('version', 'unknown')}"
        )

        changes = monitor.compare_specs(local_spec, remote_spec)
        print(f"Changes: {changes['summary']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())
