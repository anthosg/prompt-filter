#!/usr/bin/env bash

set -euo pipefail

CERT_DIR="/etc/nginx/ssl"
CERT_FILE="${CERT_DIR}/server.crt"
KEY_FILE="${CERT_DIR}/server.key"

DAYS="${SSL_CERT_DAYS:-365}"
COUNTRY="${SSL_CERT_COUNTRY:-RU}"
STATE="${SSL_CERT_STATE:-Moscow}"
LOCALITY="${SSL_CERT_LOCALITY:-Moscow}"
ORG="${SSL_CERT_ORG:-prompt-filter}"
COMMON_NAME="${SSL_CERT_COMMON_NAME:-localhost}"

mkdir -p "${CERT_DIR}"

if [[ ! -f "${CERT_FILE}" || ! -f "${KEY_FILE}" ]]; then
  echo "SSL certificate not found. Generating self-signed certificate..."

  openssl req -x509 \
    -nodes \
    -newkey rsa:2048 \
    -keyout "${KEY_FILE}" \
    -out "${CERT_FILE}" \
    -days "${DAYS}" \
    -subj "/C=${COUNTRY}/ST=${STATE}/L=${LOCALITY}/O=${ORG}/CN=${COMMON_NAME}" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

  chmod 600 "${KEY_FILE}"
  chmod 644 "${CERT_FILE}"

  echo "SSL certificate generated:"
  echo "  Certificate: ${CERT_FILE}"
  echo "  Private key: ${KEY_FILE}"
else
  echo "SSL certificate already exists."
fi

exec nginx -g "daemon off;"
