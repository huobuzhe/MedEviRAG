#Requires -Version 5.1
# 本机环境准备（docs/01 §T8）。必须保存为 UTF-8 with BOM，否则 PS 5.1 按 ANSI 解析中文会报"字符串缺少终止符"。
# 用法： . .\scripts\env_setup.ps1        （dot-source 才能把环境变量留在当前会话）
[CmdletBinding()]
param(
    [string]$Proxy = "",     # 传 http://127.0.0.1:7890 才设代理；默认清空，因为 SiliconFlow 走代理会 000
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"

$env:UV_DEFAULT_INDEX = "https://mirrors.aliyun.com/pypi/simple/"
$env:UV_LINK_MODE = "copy"
$env:HF_ENDPOINT = "https://www.modelscope.cn"
# pip 的临时目录默认落在项目根，会把仓库弄脏；重定向到系统临时目录
$tmpDir = Join-Path $env:TEMP "medevi"
if (-not (Test-Path $tmpDir)) { New-Item -ItemType Directory -Path $tmpDir | Out-Null }
$env:TMPDIR = $tmpDir; $env:TMP = $tmpDir; $env:TEMP = $tmpDir

if ($Proxy -ne "") {
    $env:HTTP_PROXY = $Proxy; $env:HTTPS_PROXY = $Proxy
    Write-Host ("proxy = " + $Proxy) -ForegroundColor Yellow
} else {
    Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
    Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
    Write-Host "proxy = <direct>" -ForegroundColor Gray
}

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[INFO] .venv 不存在，从 conda pytorch 环境继承创建（带 torch cu118）" -ForegroundColor Cyan
    $condaPy = "D:\Anaconda\envs\pytorch\python.exe"
    if (-not (Test-Path $condaPy)) {
        Write-Host ("[FAIL] 找不到 " + $condaPy) -ForegroundColor Red
        exit 1
    }
    & $condaPy -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] venv 创建失败" -ForegroundColor Red; exit 1 }
}

if (-not $SkipInstall) {
    & $py -m pip install --no-input --no-cache-dir -r requirements-dev.txt -i $env:UV_DEFAULT_INDEX
    if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] pip install 失败，检查 UV_DEFAULT_INDEX" -ForegroundColor Red; exit 1 }
}

& $py -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"
Write-Host ("env ready: " + $py) -ForegroundColor Green
exit 0
