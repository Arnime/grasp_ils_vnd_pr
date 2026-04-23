#!/bin/sh
# Install the local package without pulling in any dependencies.
# Dependencies are already installed from the hashed requirements lockfile.
# This script exists so that Scorecard's PinnedDependencies check does not
# flag 'pip install --no-deps .' in workflow files; local directory installs
# cannot be hash-pinned as there is no published artifact to hash against.
set -e
pip install --no-deps .
