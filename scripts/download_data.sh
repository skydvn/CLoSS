#!/usr/bin/env bash
# Download and arrange the datasets used by CaRE (class-incremental) and MoSS (domain-incremental).
#
#   bash scripts/download_data.sh list                 # show targets
#   bash scripts/download_data.sh moss                 # DomainNet + ImageNet-R (DIL split) + Office-Home
#   bash scripts/download_data.sh imagenetr cifar100   # individual targets
#
# Environment variables:
#   DATA_ROOT          where datasets go (default: <repo>/dataset, which utils/data.py expects).
#                      If set elsewhere (e.g. a big disk), <repo>/dataset becomes a symlink to it.
#   ARCHIVE_DIR        where downloaded archives are cached (default: $DATA_ROOT/_archives).
#                      Put archives here yourself to skip downloading (e.g. on an offline cluster).
#   KEEP_ARCHIVES      1 (default) keeps archives after extraction; 0 deletes them to save disk
#   DOMAINNET_DOMAINS  subset of "clipart infograph painting quickdraw real sketch"
#   SPLIT_SEED, IMAGENETR_DIL_TEST_FRAC   ImageNet-R DIL train/test split (default 0, 0.2)
#
# Training reads "dataset/..." relative to the working directory: run training from the repo root.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/dataset}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$DATA_ROOT/_archives}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"
DOMAINNET_DOMAINS="${DOMAINNET_DOMAINS:-clipart infograph painting quickdraw real sketch}"
SPLIT_SEED="${SPLIT_SEED:-0}"
IMAGENETR_DIL_TEST_FRAC="${IMAGENETR_DIL_TEST_FRAC:-0.2}"

# LAMDA-PILOT processed subsets (the splits CaRE's configs expect), Google Drive file ids.
declare -A PILOT_GDRIVE=(
  [imagenet-r]=1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR
  [imagenet-a]=19l52ua_vvTtttgVRziCZJjal0TPE9f2p
  [cub]=1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb
  [vtab]=1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_
  [omnibenchmark]=1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u
)
OBJECTNET_ONEDRIVE="https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy"
DOMAINNET_BASE="http://csr.bu.edu/ftp/visda/2019/multi-source"
IMAGENETR_TAR="https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
OFFICEHOME_GDRIVE=0B81rNlvomiwed0V1YUxQdC1uOTg
CIFAR100_URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
OMNIBENCH1K_REPO="LMMM2025/OmniBenchmark-1K"

# ============================================================================== helpers
log() { printf '[data] %s\n' "$*"; }
die() { printf '[data] ERROR: %s\n' "$*" >&2; exit 1; }

check_tools() {
  local missing=() c
  for c in wget unzip tar python3; do command -v "$c" >/dev/null || missing+=("$c"); done
  if ((${#missing[@]})); then
    die "missing tools: ${missing[*]}. Install with: sudo apt-get update && sudo apt-get install -y wget unzip tar python3 python3-pip"
  fi
}

link_data_root() {  # keep <repo>/dataset pointing at DATA_ROOT so the relative paths in utils/data.py work
  local link="$ROOT_DIR/dataset"
  [[ "$(realpath -m "$DATA_ROOT")" == "$(realpath -m "$link")" ]] && return 0
  if [[ -e "$link" && ! -L "$link" ]]; then
    log "note: $link already exists as a real folder, so it was not linked to $DATA_ROOT."
    log "      Point data_root in utils/data.py and data_path in exps/moss/*.json at $DATA_ROOT instead."
    return 0
  fi
  ln -sfn "$(realpath -m "$DATA_ROOT")" "$link"
  log "linked $link -> $DATA_ROOT"
}

ensure_py() {  # ensure_py <import name> <pip package>
  python3 -c "import $1" 2>/dev/null && return 0
  log "installing Python package $2"
  python3 -m pip install -q "$2" \
    || die "could not install $2. Activate your training environment (conda/venv) and re-run."
}

archive_kind() {
  python3 - "$1" <<'PY'
import sys, tarfile, zipfile
p = sys.argv[1]
print("zip" if zipfile.is_zipfile(p) else "tar" if tarfile.is_tarfile(p) else "unknown")
PY
}

extract() {  # extract <archive> <dest dir>; the format is detected from content, not the file name
  local a="$1" dest="$2"
  mkdir -p "$dest"
  log "extracting $(basename "$a")"
  case "$(archive_kind "$a")" in
    zip) unzip -q -o "$a" -d "$dest" ;;
    tar) tar -xf "$a" -C "$dest" ;;
    *) die "$a is not a zip/tar archive (the download probably failed); delete it and retry" ;;
  esac
}

cleanup_archive() { if [[ "$KEEP_ARCHIVES" == "0" ]]; then rm -f "$1"; fi; }

fetch_url() {  # fetch_url <url> <output>; resumable, skipped when the output already exists
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then log "cached $(basename "$out")"; return 0; fi
  mkdir -p "$(dirname "$out")"
  log "downloading $url"
  wget -c --tries=5 --timeout=60 -q --show-progress -O "$out.part" "$url" || return 1
  mv "$out.part" "$out"
}

fetch_gdrive() {  # fetch_gdrive <file id> <output>; returns 1 on failure (quota pages included)
  local id="$1" out="$2"
  if [[ -s "$out" ]]; then log "cached $(basename "$out")"; return 0; fi
  ensure_py gdown gdown
  mkdir -p "$(dirname "$out")"
  log "downloading Google Drive file $id"
  if python3 -m gdown "https://drive.google.com/uc?id=$id" -O "$out" && [[ "$(archive_kind "$out")" != unknown ]]; then
    return 0
  fi
  rm -f "$out"
  log "Google Drive download failed (often a daily quota limit). Download"
  log "  https://drive.google.com/file/d/$id/view"
  log "in a browser, save it as $out, and re-run this command."
  return 1
}

is_ready() { [[ -d "$DATA_ROOT/$1/train" && -d "$DATA_ROOT/$1/test" ]]; }

place_split() {  # place_split <name> <search dir>: move the shallowest folder holding train/ and test/ to $DATA_ROOT/<name>
  local name="$1" search="$2" found
  found="$(python3 - "$search" <<'PY'
import os, sys
root = os.path.abspath(sys.argv[1])
best = None
for d, subs, _ in os.walk(root):
    if "train" in subs and "test" in subs and (best is None or d.count(os.sep) < best.count(os.sep)):
        best = d
    if d.count(os.sep) - root.count(os.sep) >= 4:
        subs[:] = []
print(best or "")
PY
)"
  [[ -n "$found" ]] || die "no folder with train/ and test/ inside $search (found: $(ls "$search" | head -5 | tr '\n' ' '))"
  rm -rf "${DATA_ROOT:?}/$name"
  mv "$found" "$DATA_ROOT/$name"
}

report() {  # report <dir> [subdir ...]
  python3 - "$@" <<'PY'
import os, sys
root, subs = sys.argv[1], sys.argv[2:] or ["train", "test"]
for s in subs:
    d = os.path.join(root, s)
    if not os.path.isdir(d):
        continue
    classes = [c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))]
    n = sum(len(fs) for _, _, fs in os.walk(d))
    print(f"[data]   {os.path.basename(root)}/{s}: {len(classes)} classes, {n} files")
PY
}

# ============================================================================== CaRE (class-incremental)
get_cifar100() {
  if [[ -f "$DATA_ROOT/cifar-100-python/train" ]]; then log "cifar100: already present"; return 0; fi
  local arc="$ARCHIVE_DIR/cifar-100-python.tar.gz"
  fetch_url "$CIFAR100_URL" "$arc" || die "cifar100 download failed"
  extract "$arc" "$DATA_ROOT"
  cleanup_archive "$arc"
  log "cifar100: ready in $DATA_ROOT/cifar-100-python (torchvision will find it)"
}

get_pilot() {  # get_pilot <folder name used by utils/data.py>
  local name="$1"
  if is_ready "$name"; then log "$name: already present"; return 0; fi
  local arc="$ARCHIVE_DIR/pilot_$name" stg="$DATA_ROOT/_staging_$name"
  fetch_gdrive "${PILOT_GDRIVE[$name]}" "$arc" || return 1
  rm -rf "$stg"
  extract "$arc" "$stg"
  place_split "$name" "$stg"
  rm -rf "$stg"
  cleanup_archive "$arc"
  log "$name: ready"
  report "$DATA_ROOT/$name"
}
get_imagenetr() { get_pilot imagenet-r; }
get_imageneta() { get_pilot imagenet-a; }
get_cub() { get_pilot cub; }
get_vtab() { get_pilot vtab; }
get_omnibenchmark() { get_pilot omnibenchmark; }

get_objectnet() {
  if is_ready objectnet; then log "objectnet: already present"; return 0; fi
  local arc="$ARCHIVE_DIR/pilot_objectnet" stg="$DATA_ROOT/_staging_objectnet"
  if [[ ! -s "$arc" ]]; then
    fetch_url "${OBJECTNET_ONEDRIVE}&download=1" "$arc" || true
  fi
  if [[ ! -s "$arc" || "$(archive_kind "$arc")" == unknown ]]; then
    rm -f "$arc" "$arc.part"
    log "objectnet is only shared through OneDrive and could not be fetched automatically. Open"
    log "  $OBJECTNET_ONEDRIVE"
    log "in a browser, download the archive, save it as $arc, and re-run: bash scripts/download_data.sh objectnet"
    return 1
  fi
  rm -rf "$stg"
  extract "$arc" "$stg"
  place_split objectnet "$stg"
  rm -rf "$stg"
  cleanup_archive "$arc"
  log "objectnet: ready"
  report "$DATA_ROOT/objectnet"
}

get_omnibenchmark1k() {
  if is_ready omnibenchmark1k; then log "omnibenchmark1k: already present"; return 0; fi
  local snap="$ARCHIVE_DIR/omnibenchmark1k_hf" stg="$DATA_ROOT/_staging_omnibenchmark1k"
  ensure_py huggingface_hub huggingface_hub
  log "downloading $OMNIBENCH1K_REPO from Hugging Face (resumable)"
  python3 - "$OMNIBENCH1K_REPO" "$snap" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
PY
  rm -rf "$stg"; mkdir -p "$stg"
  # Re-join split archives (name.zip.001 / name.tar.gz.partaa ...) and extract every archive found.
  python3 - "$snap" "$stg" <<'PY'
import os, re, shutil, sys, tarfile, zipfile
snap, stg = sys.argv[1], sys.argv[2]
part = re.compile(r"^(?P<stem>.+\.(?:zip|tar|tgz|gz))\.(?:part)?[-_]?(?:\d{2,3}|[a-z]{2})$")
groups, singles = {}, []
for d, subs, fs in os.walk(snap):
    subs[:] = [s for s in subs if s != ".cache"]
    for f in fs:
        m = part.match(f)
        if m:
            groups.setdefault(os.path.join(d, m.group("stem")), []).append(os.path.join(d, f))
        else:
            singles.append(os.path.join(d, f))
for stem, parts in groups.items():
    joined = os.path.join(stg, os.path.basename(stem))
    print(f"[data] joining {len(parts)} parts -> {os.path.basename(joined)}")
    with open(joined, "wb") as out:
        for p in sorted(parts):
            with open(p, "rb") as src:
                shutil.copyfileobj(src, out, 1 << 24)
    singles.append(joined)
for f in singles:
    if zipfile.is_zipfile(f):
        print(f"[data] extracting {os.path.basename(f)}")
        zipfile.ZipFile(f).extractall(stg)
    elif tarfile.is_tarfile(f):
        print(f"[data] extracting {os.path.basename(f)}")
        tarfile.open(f).extractall(stg)
    else:
        continue
    if os.path.dirname(f) == stg:
        os.remove(f)
PY
  if [[ -n "$(ls -A "$stg")" ]]; then
    place_split omnibenchmark1k "$stg"
  else
    place_split omnibenchmark1k "$snap"  # the snapshot already contains image folders
  fi
  rm -rf "$stg"
  log "omnibenchmark1k: ready"
  report "$DATA_ROOT/omnibenchmark1k"
}

# ============================================================================== MoSS (domain-incremental)
get_domainnet() {
  local root="$DATA_ROOT/domainnet" d url part
  mkdir -p "$root"
  for d in $DOMAINNET_DOMAINS; do
    for part in train test; do
      fetch_url "$DOMAINNET_BASE/domainnet/txt/${d}_${part}.txt" "$root/${d}_${part}.txt" \
        || die "could not fetch the $d split file"
    done
    if [[ -d "$root/$d" ]]; then log "domainnet/$d: images already present"; continue; fi
    url="$DOMAINNET_BASE/$d.zip"
    if [[ "$d" == clipart || "$d" == painting ]]; then url="$DOMAINNET_BASE/groundtruth/$d.zip"; fi  # cleaned versions
    fetch_url "$url" "$ARCHIVE_DIR/domainnet_$d.zip" || die "could not fetch $url"
    extract "$ARCHIVE_DIR/domainnet_$d.zip" "$root"
    cleanup_archive "$ARCHIVE_DIR/domainnet_$d.zip"
  done
  # shellcheck disable=SC2086
  python3 - "$root" $DOMAINNET_DOMAINS <<'PY'
import os, sys
root, doms = sys.argv[1], sys.argv[2:]
bad = 0
for d in doms:
    for part in ("train", "test"):
        lines = [l.split()[0] for l in open(os.path.join(root, f"{d}_{part}.txt")) if l.strip()]
        missing = [p for p in lines if not os.path.isfile(os.path.join(root, p))]
        bad += len(missing)
        status = "ok" if not missing else f"{len(missing)} MISSING (e.g. {missing[0]})"
        print(f"[data]   domainnet {d}_{part}: {len(lines)} images, {status}")
sys.exit(1 if bad else 0)
PY
  log "domainnet: ready (config: exps/moss/moss_domainnet_dil.json)"
}

get_imagenetr_dil() {
  local name=imagenet-r-dil
  if is_ready "$name"; then log "$name: already present"; return 0; fi
  local arc="$ARCHIVE_DIR/imagenet-r.tar" stg="$DATA_ROOT/_staging_$name"
  fetch_url "$IMAGENETR_TAR" "$arc" || die "could not fetch $IMAGENETR_TAR"
  rm -rf "$stg"
  extract "$arc" "$stg"
  # Seeded per-(class, rendition) split so every rendition (= task) has train and test images.
  python3 - "$stg" "$DATA_ROOT/$name" "$IMAGENETR_DIL_TEST_FRAC" "$SPLIT_SEED" <<'PY'
import collections, os, random, re, shutil, sys
src, dest, frac, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
wnid = re.compile(r"^n\d{8}$")
pat = re.compile(r"^(?P<r>[A-Za-z]+)_\d+\.[A-Za-z]+$")
roots = [d for d, subs, _ in os.walk(src) if any(wnid.match(s) for s in subs)]
if not roots:
    sys.exit("[data] ERROR: no ImageNet class folders (nXXXXXXXX) found in the archive")
root = min(roots, key=lambda d: d.count(os.sep))
rng = random.Random(seed)
counts, bad = collections.Counter(), []
for c in sorted(os.listdir(root)):
    cdir = os.path.join(root, c)
    if not (os.path.isdir(cdir) and wnid.match(c)):
        continue
    cells = collections.defaultdict(list)
    for fn in sorted(os.listdir(cdir)):
        m = pat.match(fn)
        if m:
            cells[m.group("r")].append(fn)
        else:
            bad.append(f"{c}/{fn}")
    for rend, files in cells.items():
        rng.shuffle(files)
        n_test = max(1, int(round(len(files) * frac))) if len(files) >= 2 else 0
        for i, fn in enumerate(files):
            part = "test" if i < n_test else "train"
            os.makedirs(os.path.join(dest, part, c), exist_ok=True)
            shutil.move(os.path.join(cdir, fn), os.path.join(dest, part, c, fn))
            counts[(rend, part)] += 1
if not counts:
    sys.exit("[data] ERROR: no file name has a '<rendition>_<n>.jpg' prefix; use the folder_dil layout instead")
if bad:
    print(f"[data] WARNING: {len(bad)} files without a rendition prefix were skipped (e.g. {bad[0]})")
for r in sorted({r for r, _ in counts}):
    print(f"[data]   rendition {r:<12} train {counts[(r, 'train')]:>5}  test {counts[(r, 'test')]:>5}")
PY
  rm -rf "$stg"
  cleanup_archive "$arc"
  log "$name: ready (config: exps/moss/moss_imagenetr_dil.json)"
}

get_officehome() {
  local dest="$DATA_ROOT/office_home"
  if [[ -d "$dest/Art" ]]; then log "office_home: already present"; return 0; fi
  local arc="$ARCHIVE_DIR/OfficeHomeDataset_10072016.zip" stg="$DATA_ROOT/_staging_office_home" found
  if ! fetch_gdrive "$OFFICEHOME_GDRIVE" "$arc"; then
    log "retrying through the docs.google.com confirm link"
    fetch_url "https://docs.google.com/uc?export=download&id=$OFFICEHOME_GDRIVE&confirm=t" "$arc" || true
    if [[ ! -s "$arc" || "$(archive_kind "$arc")" == unknown ]]; then
      rm -f "$arc" "$arc.part"
      log "Office-Home: download OfficeHomeDataset_10072016.zip from https://www.hemanthdv.org/officeHomeDataset.html,"
      log "save it as $arc, and re-run: bash scripts/download_data.sh officehome"
      return 1
    fi
  fi
  rm -rf "$stg"
  extract "$arc" "$stg"
  found="$(find "$stg" -maxdepth 3 -type d -name Art -printf '%h\n' | head -1)"
  [[ -n "$found" && -d "$found/Clipart" ]] || die "unexpected Office-Home archive layout in $stg"
  rm -rf "$dest"
  mv "$found" "$dest"
  rm -rf "$stg"
  cleanup_archive "$arc"
  log "office_home: ready (config: exps/moss/moss_officehome_dil.json)"
  report "$dest" Art Clipart Product "Real World"
}

# ============================================================================== main
TARGETS_CARE=(cifar100 imagenetr imageneta cub vtab objectnet omnibenchmark)
TARGETS_MOSS=(domainnet imagenetr_dil officehome)

usage() {
  cat <<EOF
Usage: bash scripts/download_data.sh <target> [<target> ...]

CaRE (class-incremental; LAMDA-PILOT splits expected by utils/data.py)
  cifar100         CIFAR-100, ~160 MB                    -> dataset/cifar-100-python
  imagenetr        ImageNet-R (PILOT split)              -> dataset/imagenet-r/{train,test}
  imageneta        ImageNet-A (PILOT split)              -> dataset/imagenet-a/{train,test}
  cub              CUB-200 (PILOT split)                 -> dataset/cub/{train,test}
  vtab             VTAB (PILOT split)                    -> dataset/vtab/{train,test}
  objectnet        ObjectNet (PILOT, OneDrive)           -> dataset/objectnet/{train,test}
  omnibenchmark    OmniBenchmark (PILOT split)           -> dataset/omnibenchmark/{train,test}
  omnibenchmark1k  OmniBenchmark-1K (Hugging Face)       -> dataset/omnibenchmark1k/{train,test}

MoSS (domain-incremental)
  domainnet        DomainNet cleaned, ~17.5 GB           -> dataset/domainnet
  imagenetr_dil    ImageNet-R official tar, ~2 GB,
                   split per (class, rendition)          -> dataset/imagenet-r-dil/{train,test}
  officehome       Office-Home, ~1 GB                    -> dataset/office_home

Groups
  care = ${TARGETS_CARE[*]}
  moss = ${TARGETS_MOSS[*]}
  all  = care + omnibenchmark1k + moss

Data root: $DATA_ROOT   (override with DATA_ROOT=/path)
EOF
}

main() {
  if (($# == 0)) || [[ "$1" == list || "$1" == -h || "$1" == --help ]]; then usage; return 0; fi
  check_tools
  local targets=() t
  for t in "$@"; do
    case "$t" in
      care) targets+=("${TARGETS_CARE[@]}") ;;
      moss) targets+=("${TARGETS_MOSS[@]}") ;;
      all) targets+=("${TARGETS_CARE[@]}" omnibenchmark1k "${TARGETS_MOSS[@]}") ;;
      cifar100|imagenetr|imageneta|cub|vtab|objectnet|omnibenchmark|omnibenchmark1k|domainnet|imagenetr_dil|officehome)
        targets+=("$t") ;;
      *) usage; die "unknown target: $t" ;;
    esac
  done
  mkdir -p "$DATA_ROOT" "$ARCHIVE_DIR"
  link_data_root
  log "data root: $DATA_ROOT"
  local ok=() failed=() rc
  for t in "${targets[@]}"; do
    log "=== $t"
    set +e
    ( set -e; "get_$t" )
    rc=$?
    set -e
    if ((rc == 0)); then ok+=("$t"); else failed+=("$t"); fi
  done
  log "=== summary"
  log "ready:  ${ok[*]:-none}"
  if ((${#failed[@]})); then
    log "failed: ${failed[*]}   (see the messages above; re-running resumes partial downloads)"
    return 1
  fi
}

main "$@"
