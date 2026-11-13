#Requires -Version 5.1
# 启动 5 服务栈；docker 不可用时降级到 Milvus Lite（docs/01 §T8）。必须存为 UTF-8 with BOM。
# 用法： . .\scripts\start_stack.ps1
$ErrorActionPreference = "Continue"

$liteUri = "./data/indexes/milvus_lite.db"

docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] docker daemon 不可用，降级到 Milvus Lite（$liteUri）" -ForegroundColor Yellow
    $env:MILVUS_URI = $liteUri
    Write-Host "[OK] MILVUS_URI = $env:MILVUS_URI" -ForegroundColor Green
    exit 0                       # 降级是正常路径，不是失败
}

Write-Host "[OK] docker daemon 可用，启动 5 服务栈" -ForegroundColor Green
docker compose -f docker/docker-compose.yml up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] compose up 失败，降级到 Milvus Lite" -ForegroundColor Yellow
    $env:MILVUS_URI = $liteUri
    exit 0
}
docker compose -f docker/docker-compose.yml ps
$env:MILVUS_URI = "http://127.0.0.1:19530"
Write-Host "[OK] MILVUS_URI = $env:MILVUS_URI" -ForegroundColor Green
exit 0
