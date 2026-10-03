#!/usr/bin/env bash
# Baixa o binário pré-compilado do llama.cpp (Vulkan, Linux x64) para vendor/llama.cpp/<tag>
# e aponta vendor/llama.cpp/current para ele. Uso: scripts/setup-llama-cpp.sh [tag]
# Sem tag, pega a release mais recente. A tag fica registrada em cada resultado.
set -euo pipefail
cd "$(dirname "$0")/.."
# A release marcada como "latest" nem sempre é uma build (bNNNN); pega a build mais recente.
TAG="${1:-$(curl -fsSL 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20' \
  | python3 -c 'import json,sys; print(next(r["tag_name"] for r in json.load(sys.stdin) if r["tag_name"].startswith("b")))')}"
DEST="vendor/llama.cpp/$TAG"
if [[ ! -x "$DEST/llama-server" ]]; then
  mkdir -p "$DEST"
  URL="https://github.com/ggml-org/llama.cpp/releases/download/$TAG/llama-$TAG-bin-ubuntu-vulkan-x64.tar.gz"
  echo "baixando $URL"
  curl -fL "$URL" | tar -xz -C "$DEST"
  # Os tarballs recentes trazem tudo dentro de uma pasta; achata para $DEST.
  inner="$(find "$DEST" -mindepth 2 -maxdepth 3 -name llama-server -printf '%h\n' | head -1)"
  if [[ -n "$inner" && "$inner" != "$DEST" ]]; then
    mv "$inner"/* "$DEST"/ && rmdir -p --ignore-fail-on-non-empty "$inner" 2>/dev/null || true
  fi
fi
ln -sfn "$TAG" vendor/llama.cpp/current
echo "$TAG" > vendor/llama.cpp/current/TAG
"vendor/llama.cpp/current/llama-server" --version 2>&1 | tail -2
echo "ok: vendor/llama.cpp/current -> $TAG"
