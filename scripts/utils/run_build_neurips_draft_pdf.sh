#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRAFT_DIR="${REPO_ROOT}/paper/neurips2026/draft"

cd "${DRAFT_DIR}"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  exit 0
fi

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  exit 0
fi

cat >&2 <<'EOF'
No LaTeX build tool was found.

Install or load one of:
- latexmk
- pdflatex + bibtex

Then rerun:
  scripts/utils/run_build_neurips_draft_pdf.sh
EOF
exit 127
