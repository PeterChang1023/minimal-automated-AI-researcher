#!/usr/bin/env bash
# hinton rule: delete training data when done. Logs under runs/ are kept in the repo.
set -euo pipefail
rm -rf "/cache/$USER/air"
echo "removed /cache/$USER/air"
