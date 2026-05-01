#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRAFT_DIR="${REPO_ROOT}/paper/neurips2026/draft"
DEFAULT_LATEX_ENV="${HOME}/.local/share/ih-depth-dataset/latex-env"
LATEX_ENV="${LATEX_ENV_PREFIX:-${DEFAULT_LATEX_ENV}}"

if [[ -x "${LATEX_ENV}/bin/latexmk" && "${IH_LATEX_ENV_ACTIVE:-0}" != "1" ]]; then
  export IH_LATEX_ENV_ACTIVE=1
  exec mamba run -p "${LATEX_ENV}" bash "$0" "$@"
fi

cd "${DRAFT_DIR}"

cleanup_latex_intermediates() {
  rm -f \
    main.aux \
    main.bbl \
    main.bcf \
    main.blg \
    main.fdb_latexmk \
    main.fls \
    main.log \
    main.out \
    main.run.xml \
    main.synctex.gz \
    main.xdv \
    pdflatex*.fls
}

if command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex
  cleanup_latex_intermediates
  exit 0
fi

if command -v latexmk >/dev/null 2>&1; then
  latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
  cleanup_latex_intermediates
  exit 0
fi

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  cleanup_latex_intermediates
  exit 0
fi

cat >&2 <<'EOF'
No LaTeX build tool was found.

Install or load one of:
- tectonic
- latexmk
- pdflatex + bibtex

Then rerun:
  scripts/utils/run_build_neurips_draft_pdf.sh

For a user-local install, run:
  sbatch scripts/utils/submit_setup_local_latex_env.sh
EOF
exit 127
