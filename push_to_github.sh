#!/bin/bash
# Push PyBrainViewer to GitHub
# Usage: ./push_to_github.sh [commit-message]
cd "$(dirname "$0")"
if [ -n "$1" ]; then
    git add -A
    git commit -m "$1"
fi
git push -u origin main
echo "Push complete!"
