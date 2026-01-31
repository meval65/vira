#!/usr/bin/env python3
import sys
import os
import warnings

# Windows: avoid ProactorEventLoop pipe assertion errors when asyncio is used
# together with another thread (e.g. uvicorn dashboard in a daemon thread).
# Policy APIs are deprecated in 3.14+ but still required for this workaround.
if sys.platform == "win32":
    import asyncio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if policy is not None:
            asyncio.set_event_loop_policy(policy())

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.brain.brainstem import main

if __name__ == "__main__":
    main()
