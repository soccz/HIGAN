#!/bin/bash
# Compile main.tex → main.pdf via TeXLive 2026
# Usage:  bash paper/compile.sh
# Output: paper/main.pdf, with all references resolved.
set -euo pipefail

cd "$(dirname "$0")"
export PATH="/mnt/20t/texlive/2026/bin/x86_64-linux:$PATH"

echo "[1/4] pdflatex pass 1 ..."
pdflatex -interaction=nonstopmode main.tex >/tmp/p1.log 2>&1 || true

echo "[2/4] bibtex ..."
bibtex main >/tmp/b.log 2>&1 || true

echo "[3/4] pdflatex pass 2 ..."
pdflatex -interaction=nonstopmode main.tex >/tmp/p2.log 2>&1 || true

echo "[4/4] pdflatex pass 3 ..."
pdflatex -interaction=nonstopmode main.tex >/tmp/p3.log 2>&1 || true

if grep -q "^!" /tmp/p3.log; then
    echo "ERRORS:"
    grep -B1 -A2 "^!" /tmp/p3.log | head -30
    exit 1
fi
if grep -qE "Citation.*undefined" /tmp/p3.log; then
    echo "UNDEFINED CITATIONS:"
    grep -E "Citation.*undefined" /tmp/p3.log | head -10
fi
pages=$(pdfinfo main.pdf 2>/dev/null | awk '/Pages:/ {print $2}')
size=$(ls -la main.pdf | awk '{print $5}')
echo "OK — main.pdf, ${pages} pages, ${size} bytes."
