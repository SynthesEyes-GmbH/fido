# Builds Codabench Bundle.zip with the following rules:
#
#  ingestion_program/  ->  two zips, flat at the zip root:
#      ingestion_program_keypoints.zip     (ingestion.py + metadata.yaml at zip root)
#      ingestion_program_registration.zip  (ingestion.py + metadata.yaml at zip root)
#
#  scoring_program/    ->  two zips, flat at the zip root:
#      scoring_program_keypoints.zip       (scoring.py + metadata.yaml at zip root)
#      scoring_program_registration.zip    (scoring.py + metadata.yaml at zip root)
#
#  All other subfolders (input_data, reference_data, sample_submission, …)
#      are zipped flat (contents only, no wrapper folder).
#
#  pages/*.md          ->  included flat at the root of Codabench Bundle.zip
#                          (no pages/ folder in the zip)
#
#  Root files (competition.yaml, logo.png, …) are included as-is.

$bundleFolder = $PSScriptRoot
$bundleZip    = Join-Path (Split-Path $PSScriptRoot -Parent) 'Codabench Bundle.zip'
$tmp          = Join-Path $env:TEMP 'fido_bundle_staging'

if (-not (Test-Path $bundleFolder)) {
    Write-Error "Codabench Bundle folder not found at: $bundleFolder"
    exit 1
}

# ── helpers ────────────────────────────────────────────────────────────────

function New-TempDir([string]$name) {
    $path = Join-Path $tmp $name
    if (Test-Path $path) { Remove-Item $path -Recurse -Force }
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    return $path
}

function New-ProgramZip {
    param(
        [string]$SourceScript,       # full path to the source .py file
        [string]$TargetScriptName,   # name inside the zip (ingestion.py / scoring.py)
        [string]$MetadataSource,     # full path to metadata.yaml
        [string]$ZipFolderName,      # staging folder name
        [string]$ZipDestination      # full output .zip path
    )
    $stage = New-TempDir $ZipFolderName
    Copy-Item $SourceScript   (Join-Path $stage $TargetScriptName)
    Copy-Item $MetadataSource (Join-Path $stage 'metadata.yaml')
    if (Test-Path $ZipDestination) { Remove-Item $ZipDestination -Force }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $ZipDestination
    Write-Host "Created: $ZipDestination"
    Remove-Item $stage -Recurse -Force
}

# ── clean up staging root ──────────────────────────────────────────────────
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

# ── Step 1: build program zips ─────────────────────────────────────────────

$ingSrc  = Join-Path $bundleFolder 'ingestion_program'
$scrSrc  = Join-Path $bundleFolder 'scoring_program'

New-ProgramZip `
    -SourceScript     (Join-Path $ingSrc 'ingestion_keypoints.py') `
    -TargetScriptName 'ingestion.py' `
    -MetadataSource   (Join-Path $ingSrc 'metadata.yaml') `
    -ZipFolderName    'ingestion_program_keypoints' `
    -ZipDestination   (Join-Path $bundleFolder 'ingestion_program_keypoints.zip')

New-ProgramZip `
    -SourceScript     (Join-Path $ingSrc 'ingestion_registration.py') `
    -TargetScriptName 'ingestion.py' `
    -MetadataSource   (Join-Path $ingSrc 'metadata.yaml') `
    -ZipFolderName    'ingestion_program_registration' `
    -ZipDestination   (Join-Path $bundleFolder 'ingestion_program_registration.zip')

New-ProgramZip `
    -SourceScript     (Join-Path $scrSrc 'scoring_keypoints.py') `
    -TargetScriptName 'scoring.py' `
    -MetadataSource   (Join-Path $scrSrc 'metadata.yaml') `
    -ZipFolderName    'scoring_program_keypoints' `
    -ZipDestination   (Join-Path $bundleFolder 'scoring_program_keypoints.zip')

New-ProgramZip `
    -SourceScript     (Join-Path $scrSrc 'scoring_registration.py') `
    -TargetScriptName 'scoring.py' `
    -MetadataSource   (Join-Path $scrSrc 'metadata.yaml') `
    -ZipFolderName    'scoring_program_registration' `
    -ZipDestination   (Join-Path $bundleFolder 'scoring_program_registration.zip')

# ── Step 2: zip other subfolders flat (contents only) ─────────────────────

$skipFolders = @('ingestion_program', 'scoring_program', 'pages')

$otherFolders = Get-ChildItem -Path $bundleFolder -Directory |
    Where-Object { $skipFolders -notcontains $_.Name }

foreach ($folder in $otherFolders) {
    $zipPath = Join-Path $bundleFolder "$($folder.Name).zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    $contents = Get-ChildItem -Path $folder.FullName
    if ($contents) {
        Compress-Archive -Path $contents.FullName -DestinationPath $zipPath
        Write-Host "Created: $zipPath"
    }
}

# ── Step 3: build Codabench Bundle.zip ────────────────────────────────────

if (Test-Path $bundleZip) { Remove-Item $bundleZip -Force }

# Root files (competition.yaml, *.zip created above, etc.) — exclude scripts
$rootFiles = Get-ChildItem -Path $bundleFolder -File | Where-Object { $_.Extension -notin @('.ps1', '.sh') }
if ($rootFiles) {
    Compress-Archive -Path $rootFiles.FullName -DestinationPath $bundleZip
}

# pages/*.md + images — add flat (no pages/ wrapper) using a temp staging dir
$pagesStage = New-TempDir 'pages_stage'
Copy-Item (Join-Path $bundleFolder 'pages\*') $pagesStage
$pageFiles = Get-ChildItem -Path $pagesStage -File
if ($pageFiles) {
    Compress-Archive -Path $pageFiles.FullName -DestinationPath $bundleZip -Update
}
Remove-Item $pagesStage -Recurse -Force

Write-Host "Created: $bundleZip"

# ── Step 4: clean up intermediate zips from bundle folder ─────────────────

$intermediateZips = @(
    'ingestion_program_keypoints.zip', 'ingestion_program_registration.zip',
    'scoring_program_keypoints.zip',   'scoring_program_registration.zip'
) + ($otherFolders | ForEach-Object { "$($_.Name).zip" })

foreach ($name in $intermediateZips) {
    $path = Join-Path $bundleFolder $name
    if (Test-Path $path) {
        Remove-Item $path -Force
        Write-Host "Deleted: $path"
    }
}

Remove-Item $tmp -Recurse -Force
