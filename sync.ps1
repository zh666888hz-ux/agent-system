# ============================================================
# sync.ps1 - 一键同步脚本（Docker 镜像 + GitHub）
#
# 用途：代码调整后，一条命令完成「重建 Docker 镜像 + 提交 + 推送 GitHub」。
#
# 用法（在项目根目录 PowerShell 中）：
#   .\sync.ps1 "feat: 修复 xxx"     # 带提交信息同步
#   .\sync.ps1                      # 不带提交信息（默认 "chore: 同步更新"）
#
# 流程：
#   1. docker build -t react-agent .     # 重建镜像（含最新代码）
#   2. git add -A                        # 暂存所有改动
#   3. git commit                        # 提交（无改动则跳过）
#   4. git push origin main              # 推送到 GitHub
#
# 前置要求：已 git init、remote 指向 GitHub、本机已配置代理（7890）。
# ============================================================

param(
    [string]$CommitMsg = "chore: 同步更新"
)

$ErrorActionPreference = "Stop"
$git = "C:\Program Files\Git\cmd\git.exe"

Write-Host "`n========== [1/4] 重建 Docker 镜像 ==========" -ForegroundColor Cyan
docker build -t react-agent .
if ($LASTEXITCODE -ne 0) { throw "Docker 构建失败" }
Write-Host "✔ Docker 镜像已重建: react-agent:latest" -ForegroundColor Green

Write-Host "`n========== [2/4] 暂存改动 ==========" -ForegroundColor Cyan
& $git add -A
if ($LASTEXITCODE -ne 0) { throw "git add 失败" }

Write-Host "`n========== [3/4] 提交（若无改动则跳过） ==========" -ForegroundColor Cyan
$status = & $git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "✔ 无文件变更，跳过提交" -ForegroundColor Yellow
} else {
    & $git commit -m $CommitMsg
    if ($LASTEXITCODE -ne 0) { throw "git commit 失败" }
    Write-Host "✔ 已提交: $CommitMsg" -ForegroundColor Green
}

Write-Host "`n========== [4/4] 推送到 GitHub ==========" -ForegroundColor Cyan
& $git push origin main
if ($LASTEXITCODE -ne 0) { throw "git push 失败" }
Write-Host "`n✔ 全部同步完成：Docker 镜像 + GitHub 均为最新" -ForegroundColor Green
