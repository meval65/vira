#!/usr/bin/env python3
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.brain.brainstem import main

if __name__ == "__main__":
    main()
