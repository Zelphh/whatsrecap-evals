# Baixa o binário pré-compilado do llama.cpp (Vulkan, Windows x64) para vendor/llama.cpp/<tag>
# e aponta vendor/llama.cpp/current para ele (junction, não precisa de admin).
# Uso: scripts/setup-llama-cpp.ps1 [-Tag bNNNN] [-Backend vulkan|rocm-10.0|cpu]
# Sem tag, pega a build mais recente. A tag fica registrada em cada resultado.
param(
    [string]$Tag = "",
    [string]$Backend = "vulkan"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# A release marcada como "latest" nem sempre é uma build (bNNNN); pega a build mais recente.
if (-not $Tag) {
    $releases = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"
    $Tag = ($releases | Where-Object { $_.tag_name -like "b*" } | Select-Object -First 1).tag_name
}
$name = if ($Backend -eq "vulkan") { $Tag } else { "$Tag-$Backend" }
$dest = "vendor/llama.cpp/$name"
if (-not (Test-Path "$dest/llama-server.exe")) {
    New-Item -ItemType Directory -Force $dest | Out-Null
    $url = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/llama-$Tag-bin-win-$Backend-x64.zip"
    $zip = "vendor/llama.cpp/$name.zip"
    Write-Host "baixando $url"
    curl.exe -fL -o $zip $url
    if ($LASTEXITCODE -ne 0) { throw "download falhou ($url)" }
    Expand-Archive -Force $zip $dest
    Remove-Item $zip
    # Se o zip trouxer tudo dentro de uma pasta, achata para $dest.
    $inner = Get-ChildItem -Recurse -Filter llama-server.exe $dest | Select-Object -First 1
    if ($inner -and $inner.DirectoryName -ne (Resolve-Path $dest).Path) {
        Get-ChildItem $inner.DirectoryName | Move-Item -Destination $dest
    }
}
$current = "vendor/llama.cpp/current"
if (Test-Path $current) { (Get-Item $current).Delete() }
New-Item -ItemType Junction -Path $current -Target (Resolve-Path $dest).Path | Out-Null
Set-Content -Encoding ascii "$current/TAG" $name
& "$current/llama-server.exe" --version 2>&1 | Select-Object -Last 2
Write-Host "ok: $current -> $name"
