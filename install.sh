#!/bin/bash

# Force the installation using the standard Python interpreter and pip.
# The --ignore-installed flag can sometimes help resolve conflicts caused by
# the Vercel/uv environment.

echo "--- Forcing installation using system pip ---"
python -m pip install -r requirements.txt --upgrade --ignore-installed
echo "--- Installation script finished ---"
