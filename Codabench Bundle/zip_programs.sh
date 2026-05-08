#!/usr/bin/env bash
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
#  pages/*  ->  included flat at the root of Codabench Bundle.zip
#               (no pages/ folder in the zip)
#
#  Root files (competition.yaml, …) are included as-is. .sh files are excluded.

set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_ZIP="$(dirname "$BUNDLE_DIR")/Codabench Bundle.zip"
TMP_DIR="$(mktemp -d)"

trap 'rm -rf "$TMP_DIR"' EXIT

# ── helpers ────────────────────────────────────────────────────────────────

make_program_zip() {
    local source_script="$1"   # full path to source .py
    local target_name="$2"     # name inside zip (ingestion.py / scoring.py)
    local metadata_src="$3"    # full path to metadata.yaml
    local folder_name="$4"     # folder name inside the zip
    local zip_dest="$5"        # output .zip path

    local stage="$TMP_DIR/$folder_name"
    mkdir -p "$stage"
    cp "$source_script" "$stage/$target_name"
    cp "$metadata_src"  "$stage/metadata.yaml"
    rm -f "$zip_dest"
    (cd "$stage" && zip -r "$zip_dest" .)
    echo "Created: $zip_dest"
}

# ── Step 1: build program zips ─────────────────────────────────────────────

ING_SRC="$BUNDLE_DIR/ingestion_program"
SCR_SRC="$BUNDLE_DIR/scoring_program"

make_program_zip \
    "$ING_SRC/ingestion_keypoints.py"    "ingestion.py" \
    "$ING_SRC/metadata.yaml"             "ingestion_program_keypoints" \
    "$BUNDLE_DIR/ingestion_program_keypoints.zip"

make_program_zip \
    "$ING_SRC/ingestion_registration.py" "ingestion.py" \
    "$ING_SRC/metadata.yaml"             "ingestion_program_registration" \
    "$BUNDLE_DIR/ingestion_program_registration.zip"

make_program_zip \
    "$SCR_SRC/scoring_keypoints.py"      "scoring.py" \
    "$SCR_SRC/metadata.yaml"             "scoring_program_keypoints" \
    "$BUNDLE_DIR/scoring_program_keypoints.zip"

make_program_zip \
    "$SCR_SRC/scoring_registration.py"   "scoring.py" \
    "$SCR_SRC/metadata.yaml"             "scoring_program_registration" \
    "$BUNDLE_DIR/scoring_program_registration.zip"

# ── Step 2: zip other subfolders flat (contents only) ─────────────────────

SKIP_FOLDERS="ingestion_program scoring_program pages"

for folder in "$BUNDLE_DIR"/*/; do
    folder_name="$(basename "$folder")"
    # Skip the special folders
    skip=0
    for s in $SKIP_FOLDERS; do
        [[ "$folder_name" == "$s" ]] && skip=1 && break
    done
    [[ $skip -eq 1 ]] && continue

    zip_path="$BUNDLE_DIR/$folder_name.zip"
    rm -f "$zip_path"
    if [ -n "$(ls -A "$folder" 2>/dev/null)" ]; then
        (cd "$folder" && zip -r "$zip_path" .)
        echo "Created: $zip_path"
    fi
done

# ── Step 3: build Codabench Bundle.zip ────────────────────────────────────

rm -f "$BUNDLE_ZIP"

# Root files — exclude .sh, .ps1, and any directories
root_stage="$TMP_DIR/root_stage"
mkdir -p "$root_stage"
find "$BUNDLE_DIR" -maxdepth 1 -type f ! -name "*.sh" ! -name "*.ps1" \
    -exec cp {} "$root_stage/" \;

# pages/* — copy flat into root_stage (no pages/ wrapper)
cp "$BUNDLE_DIR/pages/"* "$root_stage/"

(cd "$root_stage" && zip -r "$BUNDLE_ZIP" .)
echo "Created: $BUNDLE_ZIP"

# ── Step 4: clean up intermediate zips from bundle folder ─────────────────

for zip in \
    "$BUNDLE_DIR/ingestion_program_keypoints.zip" \
    "$BUNDLE_DIR/ingestion_program_registration.zip" \
    "$BUNDLE_DIR/scoring_program_keypoints.zip" \
    "$BUNDLE_DIR/scoring_program_registration.zip"
do
    rm -f "$zip"
    echo "Deleted: $zip"
done

for folder in "$BUNDLE_DIR"/*/; do
    folder_name="$(basename "$folder")"
    skip=0
    for s in $SKIP_FOLDERS; do
        [[ "$folder_name" == "$s" ]] && skip=1 && break
    done
    [[ $skip -eq 1 ]] && continue
    zip_path="$BUNDLE_DIR/$folder_name.zip"
    if [ -f "$zip_path" ]; then
        rm -f "$zip_path"
        echo "Deleted: $zip_path"
    fi
done
