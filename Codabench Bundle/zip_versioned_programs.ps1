# Builds versioned, directly-uploadable Codabench program zips:
#
#   releases/ingestion_program_keypoints_v<N>.zip     (ingestion.py + metadata.yaml at zip root)
#   releases/ingestion_program_registration_v<N>.zip  (ingestion.py + metadata.yaml at zip root)
#   releases/scoring_program_keypoints_v<N>.zip        (scoring.py + metadata.yaml at zip root)
#   releases/scoring_program_registration_v<N>.zip     (scoring.py + metadata.yaml at zip root)
#
# Each family is versioned independently: <N> is (highest existing _vN in
# releases/ for that family) + 1, starting at 1 if none exist yet.

$bundleFolder = $PSScriptRoot
$releasesDir  = Join-Path $bundleFolder 'releases'
$tmp          = Join-Path $env:TEMP 'fido_versioned_staging'

if (-not (Test-Path $bundleFolder)) {
    Write-Error "Codabench Bundle folder not found at: $bundleFolder"
    exit 1
}

New-Item -ItemType Directory -Path $releasesDir -Force | Out-Null

if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

function Get-NextVersion {
    param([string]$BaseName)  # e.g. "ingestion_program_keypoints"

    $pattern = "^$([regex]::Escape($BaseName))_v(\d+)\.zip$"
    $existing = Get-ChildItem -Path $releasesDir -Filter "$BaseName`_v*.zip" -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.Name -match $pattern) { [int]$Matches[1] } else { $null }
        } |
        Where-Object { $_ -ne $null }

    if ($existing) { return ([int]($existing | Measure-Object -Maximum).Maximum) + 1 }
    return 1
}

function New-VersionedProgramZip {
    param(
        [string]$SourceScript,       # full path to the source .py file
        [string]$TargetScriptName,   # name inside the zip (ingestion.py / scoring.py)
        [string]$MetadataSource,     # full path to metadata.yaml
        [string]$BaseName            # e.g. "ingestion_program_keypoints"
    )

    $version = Get-NextVersion -BaseName $BaseName
    $zipName = "$BaseName`_v$version.zip"
    $zipDest = Join-Path $releasesDir $zipName

    $stage = Join-Path $tmp $BaseName
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    New-Item -ItemType Directory -Path $stage -Force | Out-Null

    Copy-Item $SourceScript   (Join-Path $stage $TargetScriptName)
    Copy-Item $MetadataSource (Join-Path $stage 'metadata.yaml')

    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zipDest
    Write-Host "Created: $zipDest"

    Remove-Item $stage -Recurse -Force
}

$ingSrc = Join-Path $bundleFolder 'ingestion_program'
$scrSrc = Join-Path $bundleFolder 'scoring_program'

New-VersionedProgramZip `
    -SourceScript     (Join-Path $ingSrc 'ingestion_keypoints.py') `
    -TargetScriptName 'ingestion.py' `
    -MetadataSource   (Join-Path $ingSrc 'metadata.yaml') `
    -BaseName         'ingestion_program_keypoints'

New-VersionedProgramZip `
    -SourceScript     (Join-Path $ingSrc 'ingestion_registration.py') `
    -TargetScriptName 'ingestion.py' `
    -MetadataSource   (Join-Path $ingSrc 'metadata.yaml') `
    -BaseName         'ingestion_program_registration'

New-VersionedProgramZip `
    -SourceScript     (Join-Path $scrSrc 'scoring_keypoints.py') `
    -TargetScriptName 'scoring.py' `
    -MetadataSource   (Join-Path $scrSrc 'metadata.yaml') `
    -BaseName         'scoring_program_keypoints'

New-VersionedProgramZip `
    -SourceScript     (Join-Path $scrSrc 'scoring_registration.py') `
    -TargetScriptName 'scoring.py' `
    -MetadataSource   (Join-Path $scrSrc 'metadata.yaml') `
    -BaseName         'scoring_program_registration'

Remove-Item $tmp -Recurse -Force
