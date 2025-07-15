#!/bin/bash

# Fix virtual environment for pyvenice project
# Run this script after closing your current shell session

cd /home/websinthe/code/pyvenice

# Clear any existing virtual environment variables
unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT

# Activate the correct virtual environment
source .venv/bin/activate

# Verify it's working
echo "VIRTUAL_ENV is now set to: $VIRTUAL_ENV"
echo "You can now run 'uv add black' without warnings"