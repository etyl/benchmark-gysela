#!/bin/bash
# Run the GYSELA compression mini-app inside the baked Singularity/Apptainer
# image (built from the same Dockerfile). The binary and PDI config live in the
# image, so only the work dir (run config + outputs) is bound. Use it as the
# Landau2X2V dataset's `launcher` parameter; the benchmark invokes it with the
# minimal contract:
#
#   landau_singularity_launch.sh <n_ranks> <config> <work_dir>
#
# The work dir is bound at /work and used as the working dir, so the absolute
# paths in the run config resolve identically inside. Singularity runs as the
# current host user by default (outputs not root-owned), so unlike docker no
# --user / --allow-run-as-root is needed. Overridable via env:
#   GYSELA_IMAGE (default docker://ghcr.io/gyselax/benchmarks/landau2x2v:latest)
#                 a docker:// URI (pulled & cached on first use) or a local .sif
#   GYSELA_BIN   (default /opt/gysela/compression_app)
#   GYSELA_PDI   (default /opt/gysela/pdi_out.yaml)
#   SINGULARITY  (default singularity; set to apptainer if that is your CLI)
set -euo pipefail

n_ranks=$1; config=$2; work_dir=$3

IMAGE="${GYSELA_IMAGE:-docker://ghcr.io/gyselax/benchmarks/landau2x2v:latest}"
BIN="${GYSELA_BIN:-/opt/gysela/compression_app}"
PDI="${GYSELA_PDI:-/opt/gysela/pdi_out.yaml}"
SINGULARITY="${SINGULARITY:-singularity}"

# --cleanenv keeps the host environment (notably any host MPI vars) from leaking
# into the container's mpirun, mirroring docker's isolation. mpirun runs inside
# the container, so n_ranks semantics match the docker launcher and no host MPI
# is required.
exec "${SINGULARITY}" exec \
    --cleanenv \
    --bind "${work_dir}:/work" \
    --pwd "/work" \
    "${IMAGE}" \
    mpirun -n "${n_ranks}" "${BIN}" "${config}" "${PDI}"
