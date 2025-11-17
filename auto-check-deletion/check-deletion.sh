#!/usr/bin/env bash

set -euo pipefail

INPUT_FILE="filtered_tests.txt"
OUTPUT_FILE="check-deletion-output.txt"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Input file '$INPUT_FILE' not found. Run filter-tests.sh first." >&2
    exit 1
fi

: > "$OUTPUT_FILE"

# Extensions we consider as potential test sources.
KNOWN_EXTENSIONS=(java kt groovy scala py rb js ts jsx tsx go cs php cpp cxx cc c hpp h mm m swift rs dart)

path_matches() {
    local path="$1"
    local package_fragment="$2"
    local class_name="$3"

    local file_name="${path##*/}"
    local base_name="${file_name%.*}"
    local extension="${file_name##*.}"

    local known=false
    for ext in "${KNOWN_EXTENSIONS[@]}"; do
        if [[ "$extension" == "$ext" ]]; then
            known=true
            break
        fi
    done

    if [[ $known == false ]]; then
        return 1
    fi

    if [[ "$base_name" != "$class_name" ]]; then
        return 1
    fi

    if [[ -n "$package_fragment" && "$path" != *"$package_fragment/$class_name."* ]]; then
        return 1
    fi

    return 0
}

extract_sha() {
    local text="$1"
    if [[ "$text" =~ ([0-9a-fA-F]{40}) ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

track_history() {
    local package_fragment="$1"
    local class_name="$2"

    local pathspec
    if [[ -n "$package_fragment" ]]; then
        pathspec="**/${package_fragment}/${class_name}.*"
    else
        pathspec="**/${class_name}.*"
    fi

    local log_output
    if ! log_output=$(git log --all --find-renames --name-status --pretty=format:'commit:%H' --reverse -- "$pathspec" 2>/dev/null); then
        echo ""
        return
    fi

    local commit=""
    local current_path=""
    local file_exists=false
    local last_existing_commit=""
    local last_existing_path=""
    local deletion_commit=""
    local deletion_path=""

    while IFS= read -r line; do
        if [[ -z "$line" ]]; then
            continue
        fi

        if [[ "$line" == commit:* ]]; then
            commit="${line#commit:}"
            continue
        fi

        local status="${line%%$'\t'*}"
        local rest="${line#*$'\t'}"

        case "$status" in
            R*)
                local old_path new_path
                IFS=$'\t' read -r old_path new_path <<< "$rest"
                if path_matches "$new_path" "$package_fragment" "$class_name" || path_matches "$old_path" "$package_fragment" "$class_name"; then
                    file_exists=true
                    current_path="$new_path"
                    last_existing_commit="$commit"
                    last_existing_path="$new_path"
                    deletion_commit=""
                    deletion_path=""
                fi
                ;;
            D*)
                local path="$rest"
                if path_matches "$path" "$package_fragment" "$class_name"; then
                    file_exists=false
                    deletion_commit="$commit"
                    deletion_path="$path"
                fi
                ;;
            A*|M*)
                local path="$rest"
                if path_matches "$path" "$package_fragment" "$class_name"; then
                    file_exists=true
                    current_path="$path"
                    last_existing_commit="$commit"
                    last_existing_path="$path"
                    deletion_commit=""
                    deletion_path=""
                fi
                ;;
        esac
    done <<< "$log_output"

    if [[ "$file_exists" == true ]]; then
        printf '%s\n%s\n%s\n' "present" "$last_existing_commit" "$current_path"
    elif [[ -n "$deletion_commit" ]]; then
        printf '%s\n%s\n%s\n%s\n%s\n' "deleted" "$deletion_commit" "$deletion_path" "$last_existing_commit" "$last_existing_path"
    else
        printf '%s\n' "unknown"
    fi
}

while IFS=',' read -r fq_test commit_info; do
    fq_test="${fq_test//\r/}"
    commit_info="${commit_info//\r/}"

    if [[ -z "$fq_test" ]]; then
        continue
    fi

    fq_test="${fq_test##\"}"
    fq_test="${fq_test%%\"}"

    commit_info="${commit_info##\"}"
    commit_info="${commit_info%%\"}"

    printf '==== Tracking %s ====%s' "$fq_test" $'\n' | tee -a "$OUTPUT_FILE"

    sha=""
    if sha=$(extract_sha "$commit_info"); then
        printf 'Referenced commit: %s%s' "$sha" $'\n' | tee -a "$OUTPUT_FILE"
    elif [[ -n "$commit_info" ]]; then
        printf 'Commit reference unavailable (raw: %s)%s' "$commit_info" $'\n' | tee -a "$OUTPUT_FILE"
    fi

    class_path="${fq_test%.*}"
    method_name="${fq_test##*.}"
    raw_class_name="${class_path##*.}"
    package_part="${class_path%.*}"
    if [[ "$package_part" == "$raw_class_name" ]]; then
        package_part=""
    fi

    file_class_name="${raw_class_name%%$*}"
    package_fragment="${package_part//./\/}"

    if [[ -z "$file_class_name" ]]; then
        printf 'Could not determine class name for %s%s' "$fq_test" $'\n' | tee -a "$OUTPUT_FILE"
        printf '%s' $'\n' | tee -a "$OUTPUT_FILE"
        continue
    fi

    printf 'Derived class name: %s (method: %s)%s' "$file_class_name" "$method_name" $'\n' | tee -a "$OUTPUT_FILE"
    if [[ -n "$package_fragment" ]]; then
        printf 'Derived package path fragment: %s%s' "$package_fragment" $'\n' | tee -a "$OUTPUT_FILE"
    else
        printf 'No package fragment detected%s' $'\n' | tee -a "$OUTPUT_FILE"
    fi

    mapfile -t history <<< "$(track_history "$package_fragment" "$file_class_name")"

    if [[ ${#history[@]} -eq 0 || -z "${history[0]}" ]]; then
        printf 'No matching history entries were found for the derived class path.%s' $'\n' | tee -a "$OUTPUT_FILE"
        printf '%s' $'\n' | tee -a "$OUTPUT_FILE"
        continue
    fi

    case "${history[0]}" in
        present)
            printf 'Status: File present in latest history. Last touched commit: %s%s' "${history[1]}" $'\n' | tee -a "$OUTPUT_FILE"
            printf 'Resolved path: %s%s' "${history[2]}" $'\n' | tee -a "$OUTPUT_FILE"
            ;;
        deleted)
            printf 'Status: File deleted in commit: %s%s' "${history[1]}" $'\n' | tee -a "$OUTPUT_FILE"
            printf 'Deleted path: %s%s' "${history[2]}" $'\n' | tee -a "$OUTPUT_FILE"
            if [[ -n "${history[3]}" ]]; then
                printf 'Last commit where file existed: %s%s' "${history[3]}" $'\n' | tee -a "$OUTPUT_FILE"
            fi
            if [[ -n "${history[4]}" ]]; then
                printf 'Last known path before deletion: %s%s' "${history[4]}" $'\n' | tee -a "$OUTPUT_FILE"
            fi
            ;;
        *)
            printf 'Status: Unable to determine file history for the derived path.%s' $'\n' | tee -a "$OUTPUT_FILE"
            ;;
    esac

    printf '%s' $'\n' | tee -a "$OUTPUT_FILE"

done < "$INPUT_FILE"

echo "Output written to $OUTPUT_FILE"
