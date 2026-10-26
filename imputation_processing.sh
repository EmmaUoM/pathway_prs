#!/usr/bin/env bash
# Genotype preparation and post-imputation processing for ADNI/AIBL.
# Requires Bash 4.4+, PLINK 1.9 and bcftools; Picard is optional for liftover.
set -Eeuo pipefail
export LC_ALL=C
trap 'printf "ERROR: line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

# Override settings with environment variables, e.g. THREADS=8 R2_MIN=0.5 bash ...
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-4}}
MEMORY_MB=${MEMORY_MB:-8192}
CHROMS=${CHROMS:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"}
VCF_PATTERN=${VCF_PATTERN:-'chr{chr}.dose.vcf.gz'}
R2_MIN=${R2_MIN:-0.3}         # Strictly greater than; use "none" to disable.
R2_TAG=${R2_TAG:-R2}          # INFO tag in the imputed VCF header.
GENO_MAX=${GENO_MAX:-0.02}
HWE_MIN=${HWE_MIN:-1e-6}
MAF_MIN=${MAF_MIN:-0.01}
OUTPUT_CHR=${OUTPUT_CHR:-26} # Use chrM for chr1/chr2/... notation.
PLINK_BIN=${PLINK_BIN:-plink}
BCFTOOLS_BIN=${BCFTOOLS_BIN:-bcftools}
PICARD_BIN=${PICARD_BIN:-picard}
PICARD_JAR=${PICARD_JAR:-}

usage() {
    cat <<'HELP'
Usage: bash genotype_pipeline.sh COMMAND ARGUMENTS

  prepare     BFILE OUT_DIR [FLIP_DIR]       Split BED and create upload VCFs.
  process     VCF_DIR OUT_DIR               Filter SNPs/R2, convert and merge.
  merge       PLINK_DIR OUT_PREFIX          Merge files named chr1, chr2, ...
  pathway     BFILE SNP_LIST OUT_PREFIX     Extract pathway SNPs after merging.
  qc          BFILE OUT_PREFIX              Apply missingness, HWE and MAF QC.
  liftover    VCF_DIR OUT_DIR CHAIN FASTA    Lift VCFs with Picard before import.
  clean-gwas  INPUT OUTPUT A1_COL A2_COL     Keep A/C/G/T SNPs in a text table.
  help                                     Show this help.

BFILE / OUT_PREFIX: filename prefix without .bed/.bim/.fam.
OUT_DIR must not already exist; existing outputs are never overwritten.
FLIP_DIR contains reviewed flip-SNP-chrN.txt lists; absent/empty lists are skipped.
Defaults: autosomes 1-22; INFO/R2 > 0.3; --geno 0.02 --hwe 1e-6 --maf 0.01.
Set CHROMS="1 2" for a subset; sex chromosomes need a separate workflow.
Set R2_MIN=none for SNP-only processing. VCF_PATTERN defaults to chr{chr}.dose.vcf.gz.
clean-gwas expects a header; set GWAS_HEADER=0 for a headerless table.
Imputation upload/download is a separate, manual step. See README.md for examples.
HELP
}

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Required executable not found: $1"; }
need_file() { [[ -f "$1" && -r "$1" && -s "$1" ]] || die "Missing, empty or unreadable file: $1"; }
need_bfile() { local ext; for ext in bed bim fam; do need_file "$1.$ext"; done; }

# Convert paths before changing directory for PLINK's whitespace-delimited merge list.
absolute_path() {
    local parent name
    parent=$(dirname -- "$1")
    name=$(basename -- "$1")
    printf '%s/%s\n' "$(cd -- "$parent" && pwd -P)" "$name"
}

new_directory() {
    [[ ! -e "$1" && ! -L "$1" ]] || die "Output directory already exists: $1. Choose a new directory."
    mkdir -p -- "$1"
}

new_prefix() {
    local ext
    mkdir -p -- "$(dirname -- "$1")"
    for ext in bed bim fam log merge-list.txt; do
        [[ ! -e "$1.$ext" && ! -L "$1.$ext" ]] || die "Output already exists: $1.$ext"
    done
}

run_plink() {
    "$PLINK_BIN" --threads "$THREADS" --memory "$MEMORY_MB" "$@"
}

vcf_path() { printf '%s/%s\n' "${1%/}" "${VCF_PATTERN//\{chr\}/$2}"; }

validate_settings() {
    local chr value
    local -A seen=()
    [[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die 'THREADS must be a positive integer.'
    [[ "$MEMORY_MB" =~ ^[1-9][0-9]*$ ]] || die 'MEMORY_MB must be a positive integer (MB).'
    read -r -a chromosomes <<< "$CHROMS"
    ((${#chromosomes[@]})) || die 'CHROMS is empty.'
    for chr in "${chromosomes[@]}"; do
        [[ "$chr" =~ ^([1-9]|1[0-9]|2[0-2])$ ]] || die "Unsupported chromosome: $chr (use 1-22)."
        [[ ! ${seen[$chr]+present} ]] || die "Repeated chromosome: $chr"
        seen[$chr]=1
    done
    [[ "$VCF_PATTERN" == *'{chr}'* ]] || die 'VCF_PATTERN must contain {chr}.'
    [[ "$R2_TAG" =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]] || die 'Invalid R2_TAG.'
    for value in "$GENO_MAX" "$HWE_MIN" "$MAF_MIN" "$R2_MIN"; do
        [[ "$value" != none ]] || continue
        [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
            && awk -v x="$value" 'BEGIN { exit !(x >= 0 && x <= 1) }' \
            || die "Invalid probability setting: $value"
    done
    [[ "$GENO_MAX" != none && "$HWE_MIN" != none && "$MAF_MIN" != none ]] \
        || die 'Only R2_MIN can be set to none.'
}

prepare() {
    local source=$1 out=$2 flips=${3:-} chr prefix
    local -a flip_args=()
    need "$PLINK_BIN"; need "$BCFTOOLS_BIN"; need_bfile "$source"
    # vcf-iid intentionally uses IID only. Reject ambiguous names before exporting.
    awk 'seen[$2]++ { exit 1 }' "$source.fam" || die 'IIDs must be unique for --recode vcf-iid.'
    if [[ -n "$flips" ]]; then
        [[ -d "$flips" ]] || die "Flip-list directory not found: $flips"
    fi
    new_directory "$out"
    mkdir -- "$out/plink" "$out/vcf"
    for chr in "${chromosomes[@]}"; do
        prefix="$out/plink/chr$chr"
        flip_args=()
        if [[ -n "$flips" && -s "$flips/flip-SNP-chr$chr.txt" ]]; then
            flip_args=(--flip "$flips/flip-SNP-chr$chr.txt")
        fi
        log "Preparing chromosome $chr"
        run_plink --bfile "$source" --chr "$chr" "${flip_args[@]}" \
            --keep-allele-order --make-bed --recode vcf-iid \
            --output-chr "$OUTPUT_CHR" --out "$prefix"
        "$BCFTOOLS_BIN" sort "$prefix.vcf" -Oz -o "$out/vcf/chr$chr.vcf.gz"
        "$BCFTOOLS_BIN" index --tbi --threads "$THREADS" "$out/vcf/chr$chr.vcf.gz"
        rm -- "$prefix.vcf"
    done
    log "Upload files are ready in $out/vcf. Select the actual input genome build on the server."
}

merge_chromosomes() {
    local directory=$1 output=$2 chr base
    local -a bims=() merge_args=()
    need "$PLINK_BIN"
    for chr in "${chromosomes[@]}"; do
        need_bfile "$directory/chr$chr"
        bims+=("$directory/chr$chr.bim")
    done
    base="${chromosomes[0]}"
    # Refuse to merge different sample sets into a partially missing genotype matrix.
    for chr in "${chromosomes[@]:1}"; do
        cmp -s <(awk '{ print $1, $2 }' "$directory/chr$base.fam") \
            <(awk '{ print $1, $2 }' "$directory/chr$chr.fam") \
            || die "Sample IDs/order differ between chr$base and chr$chr."
    done
    awk '
        $2 == "." || $2 == "0" { print "Missing variant ID at " FILENAME ":" FNR > "/dev/stderr"; exit 1 }
        seen[$2]++ { print "Duplicate variant ID: " $2 > "/dev/stderr"; exit 1 }
    ' "${bims[@]}" || die 'Resolve missing/duplicate variant IDs before merging; no variants were dropped.'
    new_prefix "$output"
    output=$(absolute_path "$output")
    directory=$(cd -- "$directory" && pwd -P)
    if ((${#chromosomes[@]} > 1)); then
        printf 'chr%s\n' "${chromosomes[@]:1}" > "$output.merge-list.txt"
        merge_args=(--merge-list "$output.merge-list.txt")
    fi
    log "Merging ${#chromosomes[@]} chromosome files"
    (
        cd -- "$directory"
        run_plink --bfile "chr$base" "${merge_args[@]}" \
            --keep-allele-order --make-bed --out "$output"
    )
    log "Merged genotype prefix: $output"
}

process_imputed() {
    local input=$1 out=$2 chr source header prefix
    local -a quality_args=()
    need "$PLINK_BIN"; need "$BCFTOOLS_BIN"
    if [[ "$R2_MIN" != none ]]; then
        quality_args=(--include "INFO/$R2_TAG > $R2_MIN")
    fi
    # Preflight all chromosome files before starting expensive work.
    for chr in "${chromosomes[@]}"; do
        source=$(vcf_path "$input" "$chr")
        need_file "$source"
        header=$("$BCFTOOLS_BIN" view --header-only "$source")
        [[ "$header" == *'##FORMAT=<ID=GT,'* ]] || die "No FORMAT/GT in $source; PLINK 1.9 requires genotype calls."
        if [[ "$R2_MIN" != none && "$header" != *"##INFO=<ID=$R2_TAG,"* ]]; then
            die "INFO/$R2_TAG is missing from $source. Set R2_TAG correctly or explicitly use R2_MIN=none."
        fi
    done
    new_directory "$out"
    mkdir -- "$out/vcf" "$out/plink"
    for chr in "${chromosomes[@]}"; do
        source=$(vcf_path "$input" "$chr")
        prefix="$out/plink/chr$chr"
        log "Filtering and converting chromosome $chr"
        "$BCFTOOLS_BIN" view --types snps --min-alleles 2 --max-alleles 2 \
            "${quality_args[@]}" --threads "$THREADS" \
            --output-type z --output-file "$out/vcf/chr$chr.filtered.vcf.gz" "$source"
        "$BCFTOOLS_BIN" index --tbi --threads "$THREADS" "$out/vcf/chr$chr.filtered.vcf.gz"
        run_plink --vcf "$out/vcf/chr$chr.filtered.vcf.gz" \
            --keep-allele-order --double-id --make-bed --out "$prefix"
    done
    merge_chromosomes "$out/plink" "$out/merged"
    log 'Processing complete. QC and pathway extraction are separate commands.'
}

extract_pathway() {
    local source=$1 snps=$2 output=$3
    need "$PLINK_BIN"; need_bfile "$source"; need_file "$snps"
    awk 'NF > 1 { exit 1 }' "$snps" || die 'SNP_LIST must contain one variant ID per line, without a header.'
    awk 'NR == FNR { if (NF) ids[$1] = 1; next } $2 in ids { found = 1 } END { exit !found }' \
        "$snps" "$source.bim" || die 'No pathway SNP IDs match this dataset.'
    new_prefix "$output"
    log 'Extracting pathway SNPs from the merged dataset'
    run_plink --bfile "$source" --extract "$snps" \
        --keep-allele-order --make-bed --out "$output"
}

quality_control() {
    local source=$1 output=$2
    need "$PLINK_BIN"; need_bfile "$source"; new_prefix "$output"
    log "Applying QC: geno=$GENO_MAX, hwe=$HWE_MIN, maf=$MAF_MIN"
    run_plink --bfile "$source" --geno "$GENO_MAX" --hwe "$HWE_MIN" \
        --maf "$MAF_MIN" --keep-allele-order --make-bed --out "$output"
}

liftover_vcfs() {
    local input=$1 out=$2 chain=$3 fasta=$4 chr source dictionary
    local -a picard_command=()
    need_file "$chain"; need_file "$fasta"
    [[ "$fasta" != *.gz ]] || die 'Use an uncompressed target .fa or .fasta reference.'
    dictionary="${fasta%.*}.dict"
    need_file "$fasta.fai"; need_file "$dictionary"
    if [[ -n "$PICARD_JAR" ]]; then
        need java; need_file "$PICARD_JAR"
        picard_command=(java "-Xmx${MEMORY_MB}m" -jar "$PICARD_JAR")
    else
        need "$PICARD_BIN"
        picard_command=("$PICARD_BIN")
    fi
    for chr in "${chromosomes[@]}"; do need_file "$(vcf_path "$input" "$chr")"; done
    new_directory "$out"
    mkdir -- "$out/rejected"
    for chr in "${chromosomes[@]}"; do
        source=$(vcf_path "$input" "$chr")
        log "Lifting chromosome $chr; inspect rejected variants before downstream analysis"
        "${picard_command[@]}" LiftoverVcf \
            "INPUT=$source" "OUTPUT=$out/chr$chr.dose.vcf.gz" \
            "CHAIN=$chain" "REFERENCE_SEQUENCE=$fasta" \
            "REJECT=$out/rejected/chr$chr.rejected.vcf.gz" \
            WRITE_ORIGINAL_POSITION=true RECOVER_SWAPPED_REF_ALT=false CREATE_INDEX=true
    done
    log "Lifted VCFs: $out (named chrN.dose.vcf.gz). Review Picard's rejection counts."
}

clean_gwas() {
    local input=$1 output=$2 a1=$3 a2=$4 temporary header=${GWAS_HEADER:-1}
    need_file "$input"
    [[ "$a1" =~ ^[1-9][0-9]*$ && "$a2" =~ ^[1-9][0-9]*$ && "$a1" != "$a2" ]] \
        || die 'Allele columns must be distinct positive integers, counted from 1.'
    [[ "$header" == 0 || "$header" == 1 ]] || die 'GWAS_HEADER must be 0 or 1.'
    [[ ! -e "$output" && ! -L "$output" ]] || die "Output already exists: $output"
    mkdir -p -- "$(dirname -- "$output")"
    temporary=$(mktemp "$output.tmp.XXXXXX")

    if awk -v a="$a1" -v b="$a2" -v header="$header" '
        NR == 1 && header { print; next }
        NF == 0 { next }
        NF < a || NF < b { print "Missing allele columns on line " NR > "/dev/stderr"; bad = 1; exit 1 }
        toupper($a) ~ /^[ACGT]$/ && toupper($b) ~ /^[ACGT]$/ && toupper($a) != toupper($b) { print; kept++; next }
        { removed++ }
        END { if (!bad) printf "GWAS rows: kept=%d, removed=%d\n", kept, removed > "/dev/stderr" }
    ' "$input" > "$temporary"; then
        mv -- "$temporary" "$output"
    else
        rm -- "$temporary"
        die 'GWAS filtering failed; no final output was written.'
    fi
}

main() {
    local command=${1:-help}
    (($# == 0)) || shift
    case "$command" in help|-h|--help) usage; return ;; esac
    ((BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4))) \
        || die 'Bash 4.4 or newer is required (Linux/HPC recommended).'
    validate_settings
    case "$command" in
        prepare)    [[ $# == 2 || $# == 3 ]] || die 'Usage: prepare BFILE OUT_DIR [FLIP_DIR]'; prepare "$@" ;;
        process)    [[ $# == 2 ]] || die 'Usage: process VCF_DIR OUT_DIR'; process_imputed "$@" ;;
        merge)      [[ $# == 2 ]] || die 'Usage: merge PLINK_DIR OUT_PREFIX'; merge_chromosomes "$@" ;;
        pathway)    [[ $# == 3 ]] || die 'Usage: pathway BFILE SNP_LIST OUT_PREFIX'; extract_pathway "$@" ;;
        qc)         [[ $# == 2 ]] || die 'Usage: qc BFILE OUT_PREFIX'; quality_control "$@" ;;
        liftover)   [[ $# == 4 ]] || die 'Usage: liftover VCF_DIR OUT_DIR CHAIN FASTA'; liftover_vcfs "$@" ;;
        clean-gwas) [[ $# == 4 ]] || die 'Usage: clean-gwas INPUT OUTPUT A1_COL A2_COL'; clean_gwas "$@" ;;
        *) die "Unknown command: $command. Run with --help." ;;
    esac
}

main "$@"
