#!/bin/bash
cd "$(dirname "$0")"
echo "Compiling PyBrainViewer User Manual..."
xelatex -interaction=nonstopmode manual.tex && \
xelatex -interaction=nonstopmode manual.tex
echo "Done. Output: manual.pdf"
