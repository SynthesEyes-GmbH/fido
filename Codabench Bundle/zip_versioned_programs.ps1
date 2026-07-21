# Builds directly-uploadable Codabench program zips, overwriting each run:
#
#   releases/ingestion_program_keypoints.zip      (ingestion.py + metadata.yaml at zip root)
#   releases/ingestion_program_registration.zip   (ingestion.py + metadata.yaml at zip root)
#   releases/scoring_program_keypoints.zip        (scoring.py + metadata.yaml at zip root)
#   releases/scoring_program_registration.zip     (scoring.py + metadata.yaml at zip root)

$bundleFolder = $PSScriptRoot
$releasesDir  = Join-Path $bundleFolder 'releases'
$tmp          = Join-Path $env:TEMP 'fido_program_staging'

if (-not (Test-Path $bundleFolder)) {
    Write-Error "Codabench Bundle folder not found at: $bundleFolder"
    exit 1
}

New-Item -ItemType Directory -Path $releasesDir -Force | Out-Null

if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

function New-ProgramZip {
    param(
        [string]$SourceScript,       # full path to the source .py file
        [string]$TargetScriptName,   # name inside the zip (ingestion.py / scoring.py)
        [string]$MetadataSource,     # full path to metadata.yaml
        [string]$BaseName            # e.g. "ingestion_program_keypoints"
    )

    if (-not (Test-Path $SourceScript)) {
        Write-Error "Source script not found: $SourceScript"
        return
    }
    if (-not (Test-Path $MetadataSource)) {
        Write-Error "Metadata file not found: $MetadataSource"
        return
    }

    $zipDest = Join-Path $releasesDir "$BaseName.zip"

    $stage = Join-Path $tmp $BaseName
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    New-Item -ItemType Directory -Path $stage -Force | Out-Null

    Copy-Item -Path $SourceScript   -Destination (Join-Path $stage $TargetScriptName) -Force
    Copy-Item -Path $MetadataSource -Destination (Join-Path $stage 'metadata.yaml') -Force

    if (Test-Path $zipDest) { Remove-Item $zipDest -Force }
    Compress-Archive -Path "$stage\*" -DestinationPath $zipDest -Force
    Write-Host "Created: $zipDest"

    Remove-Item $stage -Recurse -Force
}

$ingSrc = Join-Path $bundleFolder 'ingestion_program'
$scrSrc = Join-Path $bundleFolder 'scoring_program'

New-ProgramZip `
    -SourceScript     (Join-Path $ingSrc 'ingestion_keypoints.py') `
    -TargetScriptName 'ingestion.py' `
    -MetadataSource   (Join-Path $ingSrc 'metadata.yaml') `
    -BaseName         'ingestion_program_keypoints'`

New-ProgramZip `
    -SourceScript     (Join-Path $ingSrc 'ingestion_registration.py') `
    -TargetScriptName 'ingestion.py' `
    -MetadataSource   (Join-Path $ingSrc 'metadata.yaml') `
    -BaseName         'ingestion_program_registration'

New-ProgramZip `
    -SourceScript     (Join-Path $scrSrc 'scoring_keypoints.py') `
    -TargetScriptName 'scoring.py' `
    -MetadataSource   (Join-Path $scrSrc 'metadata.yaml') `
    -BaseName         'scoring_program_keypoints'

New-ProgramZip `
    -SourceScript     (Join-Path $scrSrc 'scoring_registration.py') `
    -TargetScriptName 'scoring.py' `
    -MetadataSource   (Join-Path $scrSrc 'metadata.yaml') `
    -BaseName         'scoring_program_registration'

Remove-Item $tmp -Recurse -Force
