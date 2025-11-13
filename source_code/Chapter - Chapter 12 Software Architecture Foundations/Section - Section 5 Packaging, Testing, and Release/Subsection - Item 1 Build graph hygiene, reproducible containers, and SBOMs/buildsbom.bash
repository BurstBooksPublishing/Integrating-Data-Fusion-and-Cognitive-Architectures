#!/usr/bin/env bash
set -euo pipefail
IMAGE_NAME="registry.example.com/fusion-cog/node"
COMMIT_SHA=$(git rev-parse --short=8 HEAD)         # commit provenance
BASE_DIGEST="docker.io/library/debian@sha256:..." # pinned base image digest
OUT_TAG="${IMAGE_NAME}:${COMMIT_SHA}"

# Build with BuildKit for reproducibility; pass fixed build-args and inline cache.
DOCKER_BUILDKIT=1 docker buildx build \
  --builder default \
  --pull --provenance=false \
  --build-arg BASE_IMAGE=${BASE_DIGEST} \
  --label "org.opencontainers.image.revision=${COMMIT_SHA}" \
  --label "org.opencontainers.image.source=$(git config --get remote.origin.url)" \
  --target runtime \
  --cache-to=type=inline \
  --cache-from=type=local,src=.buildcache \
  -t "${OUT_TAG}" . --push                      # push immutable image

# Generate SBOM (CycloneDX) for image layers and packages.
syft "${OUT_TAG}" -o cyclonedx-json > sbom-${COMMIT_SHA}.json  # syft required

# Sign image and SBOM (requires cosign key in COSIGN_KEY env).
cosign sign --key "$COSIGN_KEY" "${OUT_TAG}"                   # image signature
cosign sign --key "$COSIGN_KEY" sbom-${COMMIT_SHA}.json       # SBOM signature

# Upload artifacts to artifact registry (example uses curl to an internal store).
curl -X POST -H "Content-Type: application/json" \
  --data-binary @sbom-${COMMIT_SHA}.json \
  "https://artifacts.example.com/sbom/${IMAGE_NAME}/${COMMIT_SHA}"