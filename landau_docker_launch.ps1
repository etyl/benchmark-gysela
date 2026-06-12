#requires -version 5
# Windows equivalent of landau_docker_launch.sh: run the GYSELA compression
# mini-app inside the baked docker image (see Dockerfile). The binary and PDI
# config live in the image, so only the work dir (run config + outputs) is
# mounted. Use it as the Landau2X2V dataset's `launcher` parameter; the
# benchmark invokes it with the minimal contract:
#
#   landau_docker_launch.ps1 <n_ranks> <config> <work_dir>
#
#   GYSELA_IMAGE (default ghcr.io/gyselax/benchmarks/landau2x2v:latest)
#   GYSELA_BIN   (default /opt/gysela/compression_app)
#   GYSELA_PDI   (default /opt/gysela/pdi_out.yaml)
#
# Note: unlike the POSIX script there is no `--user $(id -u):$(id -g)`. On
# Docker Desktop for Windows the bind mount maps host file ownership for you,
# so the container runs mpirun as root (with --allow-run-as-root).
param(
    [Parameter(Mandatory = $true)][string]$NRanks,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$WorkDir
)
$ErrorActionPreference = "Stop"

$Image = if ($env:GYSELA_IMAGE) { $env:GYSELA_IMAGE } else { "ghcr.io/gyselax/benchmarks/landau2x2v:latest" }
$Bin   = if ($env:GYSELA_BIN)   { $env:GYSELA_BIN }   else { "/opt/gysela/compression_app" }
$Pdi   = if ($env:GYSELA_PDI)   { $env:GYSELA_PDI }   else { "/opt/gysela/pdi_out.yaml" }

# Docker Desktop wants a forward-slash absolute path for bind mounts.
$MountDir = (Resolve-Path -LiteralPath $WorkDir).Path -replace '\\', '/'

docker run --rm `
    -e OMP_PROC_BIND=spread `
    -e OMP_PLACES=threads `
    -v "${MountDir}:/work" `
    --workdir "/work" `
    $Image `
    mpirun --allow-run-as-root -n $NRanks --bind-to none `
        -x OMP_PROC_BIND -x OMP_PLACES -x OMP_NUM_THREADS `
        $Bin $Config $Pdi

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
