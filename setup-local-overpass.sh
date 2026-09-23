#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.overpass.yml"
ENV_FILE="${SCRIPT_DIR}/.env"
DEFAULT_OVERPASS_URL="http://localhost:12345/api/interpreter"

# The compose file declares the database volume `external: true`, so that an
# import survives a compose project rename or teardown. Compose will not create
# an external volume -- it refuses to start when one is missing -- so this
# script has to make it first, or the single command this script exists to
# provide fails on a clean machine. `docker volume create` is idempotent, so
# re-running costs nothing and never touches an existing database.
OVERPASS_DB_VOLUME="sih26137-overpass-db"

region="central-zone"
extract_url=""
dry_run="false"

usage() {
  cat <<'USAGE'
Usage: ./setup-local-overpass.sh [--region <name>] [--extract-url <url>] [--dry-run]

Examples:
  ./setup-local-overpass.sh --region central-zone
  ./setup-local-overpass.sh --region india
  ./setup-local-overpass.sh --extract-url https://download.geofabrik.de/asia/india/central-zone-latest.osm.pbf

Notes:
  - --region india uses the full-country extract (slowest import).
  - Any other --region value maps to:
      https://download.geofabrik.de/asia/india/<region>-latest.osm.pbf
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      [[ $# -lt 2 ]] && { echo "Missing value for --region" >&2; exit 1; }
      region="$2"
      shift 2
      ;;
    --extract-url)
      [[ $# -lt 2 ]] && { echo "Missing value for --extract-url" >&2; exit 1; }
      extract_url="$2"
      shift 2
      ;;
    --dry-run)
      dry_run="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

normalize() {
  local value="$1"
  value="${value,,}"
  value="${value// /-}"
  value="${value//_/-}"
  echo "${value}"
}

if [[ -z "${extract_url}" ]]; then
  normalized_region="$(normalize "${region}")"
  case "${normalized_region}" in
    india|india-wide|full-india)
      extract_url="https://download.geofabrik.de/asia/india-latest.osm.pbf"
      runtime_hint="~2+ hours for first import"
      ;;
    *)
      slug="${normalized_region%-latest}"
      slug="${slug%.osm.pbf}"
      extract_url="https://download.geofabrik.de/asia/india/${slug}-latest.osm.pbf"
      runtime_hint="~15-40 min for most state/zone extracts (minutes for very small city extracts)"
      ;;
  esac
else
  runtime_hint="Depends on extract size: minutes (city) to 2+ hours (India-wide)"
fi

set_env_var() {
  local key="$1"
  local value="$2"
  if [[ -f "${ENV_FILE}" ]]; then
    if grep -qE "^${key}=" "${ENV_FILE}"; then
      sed -i "s|^${key}=.*|${key}=${value}|" "${ENV_FILE}"
    else
      printf '\n%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
    fi
  else
    printf '%s=%s\n' "${key}" "${value}" > "${ENV_FILE}"
  fi
}

echo "Using extract: ${extract_url}"
echo "Expected first import time: ${runtime_hint}"
echo "Compose file: ${COMPOSE_FILE}"

if [[ "${dry_run}" == "true" ]]; then
  echo "[dry-run] Would write OSM_EXTRACT_URL and OVERPASS_URL to ${ENV_FILE}"
  echo "[dry-run] Would run: docker volume create ${OVERPASS_DB_VOLUME}"
  echo "[dry-run] Would run: docker compose -f ${COMPOSE_FILE} up -d"
  exit 0
fi

set_env_var "OSM_EXTRACT_URL" "${extract_url}"
set_env_var "OVERPASS_URL" "${DEFAULT_OVERPASS_URL}"

if docker volume inspect "${OVERPASS_DB_VOLUME}" >/dev/null 2>&1; then
  echo "Reusing existing database volume ${OVERPASS_DB_VOLUME} (no re-import)."
else
  echo "Creating persistent database volume ${OVERPASS_DB_VOLUME}."
  docker volume create "${OVERPASS_DB_VOLUME}" >/dev/null
fi

docker compose -f "${COMPOSE_FILE}" up -d

echo
echo "Overpass container started. Import runs in background."
echo "Follow progress: docker compose -f ${COMPOSE_FILE} logs -f overpass"
echo "Verify endpoint: python ${SCRIPT_DIR}/scripts/check_overpass.py --compare"
