#!/bin/bash
#SBATCH --job-name=ih_latex_setup
#SBATCH --output=logs/out/%j_ih_latex_setup.out
#SBATCH --error=logs/err/%j_ih_latex_setup.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/out logs/err

LATEX_ENV_PREFIX="${LATEX_ENV_PREFIX:-${HOME}/.local/share/ih-depth-dataset/latex-env}"

if [[ -x "${LATEX_ENV_PREFIX}/bin/latexmk" && -x "${LATEX_ENV_PREFIX}/bin/pdflatex" ]]; then
  echo "Local LaTeX environment already exists: ${LATEX_ENV_PREFIX}"
else
  echo "Creating user-local LaTeX environment: ${LATEX_ENV_PREFIX}"
  mamba create -y -p "${LATEX_ENV_PREFIX}" -c conda-forge tectonic latexmk texlive-core
fi

if [[ ! -x "${LATEX_ENV_PREFIX}/bin/tectonic" ]]; then
  echo "Installing tectonic into local LaTeX environment: ${LATEX_ENV_PREFIX}"
  mamba install -y -p "${LATEX_ENV_PREFIX}" -c conda-forge tectonic
fi

export LATEX_ENV_PREFIX
scripts/utils/run_build_neurips_draft_pdf.sh
