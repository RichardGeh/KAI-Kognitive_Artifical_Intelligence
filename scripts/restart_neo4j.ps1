<#
.SYNOPSIS
    Restart Neo4j Desktop and wait for database availability.

.DESCRIPTION
    This script stops the Neo4j Desktop process, waits for graceful shutdown,
    restarts Neo4j Desktop, and polls the bolt connection until the database
    is available or timeout is reached.

    Exit Codes:
    - 0: Success - Neo4j restarted and database is available
    - 1: Failure - Could not restart or connect to Neo4j

.EXAMPLE
    .\restart_neo4j.ps1
    Restarts Neo4j Desktop with default settings.

.EXAMPLE
    .\restart_neo4j.ps1 -MaxWaitSeconds 120
    Restarts Neo4j Desktop and waits up to 120 seconds for availability.

.NOTES
    Author: KAI Project
    Requires: Neo4j Desktop installed at specified path
    Date: 2025-12-25
#>

param(
    [string]$Neo4jPath = "C:\Users\gehri\AppData\Local\Programs\neo4j-desktop\Neo4j Desktop.exe",
    [int]$ShutdownWaitSeconds = 10,
    [int]$MaxWaitSeconds = 60,
    [int]$PollIntervalSeconds = 3,
    [string]$BoltUri = "bolt://127.0.0.1:7687"
)

# ASCII-safe output for Windows console
$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message, [string]$Type = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    switch ($Type) {
        "INFO"    { Write-Host "[$timestamp] [INFO] $Message" }
        "SUCCESS" { Write-Host "[$timestamp] [OK] $Message" -ForegroundColor Green }
        "ERROR"   { Write-Host "[$timestamp] [ERROR] $Message" -ForegroundColor Red }
        "WARN"    { Write-Host "[$timestamp] [WARN] $Message" -ForegroundColor Yellow }
    }
}

function Test-Neo4jConnection {
    <#
    .SYNOPSIS
        Test if Neo4j is responding on the bolt port.
    #>
    param([string]$Uri)

    try {
        # Parse host and port from bolt URI
        $uriParts = $Uri -replace "bolt://", ""
        $hostPort = $uriParts -split ":"
        $targetHost = $hostPort[0]  # Avoid conflict with PowerShell $host automatic variable
        $port = if ($hostPort.Length -gt 1) { [int]$hostPort[1] } else { 7687 }

        # Try TCP connection to bolt port
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $tcpClient.BeginConnect($targetHost, $port, $null, $null)
        $waitResult = $asyncResult.AsyncWaitHandle.WaitOne(2000, $false)

        if ($waitResult -and $tcpClient.Connected) {
            $tcpClient.Close()
            return $true
        }

        $tcpClient.Close()
        return $false
    }
    catch {
        return $false
    }
}

function Stop-Neo4jDesktop {
    <#
    .SYNOPSIS
        Stop Neo4j Desktop process gracefully.
    #>
    Write-Status "Stopping Neo4j Desktop process..."

    $processes = Get-Process -Name "Neo4j Desktop" -ErrorAction SilentlyContinue

    if (-not $processes) {
        Write-Status "Neo4j Desktop is not running." "WARN"
        return $true
    }

    foreach ($proc in $processes) {
        try {
            # Try graceful stop first
            $proc.CloseMainWindow() | Out-Null
        }
        catch {
            Write-Status "Could not send close signal to process $($proc.Id)" "WARN"
        }
    }

    # Wait for graceful shutdown
    Write-Status "Waiting $ShutdownWaitSeconds seconds for graceful shutdown..."
    Start-Sleep -Seconds $ShutdownWaitSeconds

    # Check if still running and force kill if necessary
    $remainingProcesses = Get-Process -Name "Neo4j Desktop" -ErrorAction SilentlyContinue
    if ($remainingProcesses) {
        Write-Status "Force stopping remaining Neo4j Desktop processes..." "WARN"
        $remainingProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }

    # Verify stopped
    $finalCheck = Get-Process -Name "Neo4j Desktop" -ErrorAction SilentlyContinue
    if ($finalCheck) {
        Write-Status "Could not stop Neo4j Desktop processes." "ERROR"
        return $false
    }

    Write-Status "Neo4j Desktop stopped." "SUCCESS"
    return $true
}

function Start-Neo4jDesktop {
    <#
    .SYNOPSIS
        Start Neo4j Desktop application.
    #>

    if (-not (Test-Path $Neo4jPath)) {
        Write-Status "Neo4j Desktop not found at: $Neo4jPath" "ERROR"
        return $false
    }

    Write-Status "Starting Neo4j Desktop from: $Neo4jPath"

    try {
        Start-Process -FilePath $Neo4jPath -WindowStyle Normal
        Write-Status "Neo4j Desktop process started." "SUCCESS"
        return $true
    }
    catch {
        Write-Status "Failed to start Neo4j Desktop: $_" "ERROR"
        return $false
    }
}

function Wait-Neo4jAvailable {
    <#
    .SYNOPSIS
        Poll bolt connection until Neo4j is available or timeout.
    #>
    param([string]$Uri, [int]$MaxSeconds, [int]$IntervalSeconds)

    Write-Status "Waiting for Neo4j database to become available (max $MaxSeconds seconds)..."

    $elapsed = 0
    while ($elapsed -lt $MaxSeconds) {
        if (Test-Neo4jConnection -Uri $Uri) {
            Write-Status "Neo4j is responding on $Uri" "SUCCESS"
            return $true
        }

        Write-Status "Database not yet available, waiting... ($elapsed/$MaxSeconds seconds)"
        Start-Sleep -Seconds $IntervalSeconds
        $elapsed += $IntervalSeconds
    }

    Write-Status "Timeout waiting for Neo4j to become available after $MaxSeconds seconds." "ERROR"
    return $false
}

# Main execution
Write-Status "=========================================="
Write-Status "Neo4j Desktop Restart Script"
Write-Status "=========================================="
Write-Status "Neo4j Path: $Neo4jPath"
Write-Status "Bolt URI: $BoltUri"
Write-Status "Max wait time: $MaxWaitSeconds seconds"
Write-Status ""

# Step 1: Stop Neo4j Desktop
if (-not (Stop-Neo4jDesktop)) {
    Write-Status "Failed to stop Neo4j Desktop." "ERROR"
    exit 1
}

# Brief pause before restart
Start-Sleep -Seconds 2

# Step 2: Start Neo4j Desktop
if (-not (Start-Neo4jDesktop)) {
    Write-Status "Failed to start Neo4j Desktop." "ERROR"
    exit 1
}

# Step 3: Wait for database availability
# Initial delay to allow Neo4j Desktop UI to initialize
Write-Status "Waiting 10 seconds for Neo4j Desktop to initialize..."
Start-Sleep -Seconds 10

if (-not (Wait-Neo4jAvailable -Uri $BoltUri -MaxSeconds $MaxWaitSeconds -IntervalSeconds $PollIntervalSeconds)) {
    Write-Status "Neo4j restart completed but database may not be fully available." "WARN"
    Write-Status "You may need to manually start the database in Neo4j Desktop." "WARN"
    exit 1
}

Write-Status ""
Write-Status "=========================================="
Write-Status "Neo4j Desktop restarted successfully!" "SUCCESS"
Write-Status "=========================================="
exit 0
