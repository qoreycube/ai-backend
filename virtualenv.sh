#!/bin/bash
# This script creates a Python virtual environment and installs dependencies from requirements.txt

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
