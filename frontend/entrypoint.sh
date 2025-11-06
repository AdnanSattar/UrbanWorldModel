#!/bin/sh

# Ensure node_modules exists and has content by installing dependencies if needed
if [ ! -d "node_modules" ] || [ -z "$(ls -A node_modules 2>/dev/null)" ]; then
  echo "Installing dependencies..."
  npm install
else
  echo "Dependencies already installed."
fi

# Execute the command passed to the container
exec "$@"

