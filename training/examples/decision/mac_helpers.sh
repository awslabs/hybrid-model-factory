# Functions that run_pipeline_mac.sh and finetune_mac.sh share, sourced by both (not run on its own). They run from
# training/, in HMF's environment with the mac extra (pip install -e ".[mac]").

PY="${PY:-python}"
STRANDS_DIR="${STRANDS_DIR:-strands-decider}"
# The Strands Decider version whose data recipe, checksums, and teacher answers the scripts use
STRANDS_COMMIT=632487dcd6120c42a127b95701a97870c49b6f67

# step TITLE: prints a step's title
step() { printf '\n========== %s ==========\n' "$*"; }

# derive FILE NAME KEY=VALUE...: writes FILE with the overrides to a temporary file and prints its path. A key's
# parts are separated by dots, and * stands for every item of a list (e.g., benchmarks.*.limit=50).
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
derive() {
    local file=$1 name=$2
    shift 2
    "$PY" - "$file" "$WORK/$name" "$@" <<'EOF'
import sys

import yaml

source, target, *overrides = sys.argv[1:]
with open(source) as f:
    config = yaml.safe_load(f)
for override in overrides:
    key, value = override.split("=", 1)
    *parents, last = key.split(".")
    nodes = [config]
    for part in parents:
        nodes = [child for node in nodes for child in (node if part == "*" else [node[part]])]
    for node in nodes:
        node[last] = yaml.safe_load(value)
with open(target, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
EOF
    echo "$WORK/$name"
}

# require_mlx: stops, with the commands that fix it, unless the Python in PY has HMF and MLX installed
require_mlx() {
    local python where
    python="$(command -v "$PY" || true)"
    where="for $python, which is probably not HMF's environment"
    [ -n "$python" ] || where="here: there is no $PY command, which HMF's environment provides once it is active"
    if ! "$PY" -c "import hmf" 2>/dev/null; then
        cat >&2 <<EOF
HMF is not installed $where.
Activate HMF's environment first, or create one from training/ (Python 3.11 or newer):

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[mac]"

Then run this script again in the same shell (examples/decision/README.md, "Setting Up a Mac").
EOF
        exit 1
    fi
    if ! "$PY" -c "import mlx_lm" 2>/dev/null; then
        cat >&2 <<EOF
HMF is installed for $python, but MLX is not. From training/, install HMF's mac extra in the same environment:

    pip install -e ".[mac]"
EOF
        exit 1
    fi
}

# strands_data FILE...: makes Strands Decider's data available in data/strands-decider, with the files given (paths
# within its data/ folder, e.g., holdout_v5_norule.jsonl). It clones Strands Decider into STRANDS_DIR at
# STRANDS_COMMIT unless the clone exists, runs the steps of its training/recipe.sh that build the missing files, and
# links its data/ folder to data/strands-decider.
strands_data() {
    if [ ! -d "$STRANDS_DIR/.git" ]; then
        git clone https://github.com/strands-labs/strands-decider "$STRANDS_DIR"
    fi
    if [ "$(git -C "$STRANDS_DIR" rev-parse HEAD)" != "$STRANDS_COMMIT" ]; then
        git -C "$STRANDS_DIR" fetch --quiet origin
        git -C "$STRANDS_DIR" checkout --quiet "$STRANDS_COMMIT"
    fi
    # A download that is not a zip (e.g., Google Drive's "Quota exceeded" page for MuSiQue) is removed, since
    # recipe.sh skips the downloads that exist, and would otherwise fail on its checksum again
    local zip path name
    for zip in contract-nli.zip musique_data_v1.0.zip; do
        path="$STRANDS_DIR/data/raw/$zip"
        if [ -f "$path" ] && [ "$(head -c 2 "$path")" != "PK" ]; then
            echo "Removing $path, which is not a zip file (its download failed, e.g., on a quota). Run again later." >&2
            rm "$path"
        fi
    done
    # The recipe steps that build the missing files, in the recipe's order: build (the v5 corpus and the held-out
    # file), fetch (the public downloads, for multistep and adequacy), multistep, generated, and adequacy
    local build="" fetch="" multistep="" generated="" adequacy=""
    for name; do
        [ -f "$STRANDS_DIR/data/$name" ] && continue
        case "$name" in
            train_v5*|holdout_v5*) build=1 ;;
            multistep_v14*) fetch=1; multistep=1 ;;
            generated_v1*) generated=1 ;;
            adequacy_hs2*) fetch=1; adequacy=1 ;;
            adequacy_gen*) adequacy=1 ;;
            *) echo "$STRANDS_DIR/data/$name is missing, and no recipe step builds it." >&2; exit 1 ;;
        esac
    done
    local steps=()
    [ -n "$build" ] && steps+=(build)
    [ -n "$fetch" ] && steps+=(fetch)
    [ -n "$multistep" ] && steps+=(multistep)
    [ -n "$generated" ] && steps+=(generated)
    [ -n "$adequacy" ] && steps+=(adequacy)
    if [ ${#steps[@]} -gt 0 ]; then
        echo "Building Strands Decider's data: ${steps[*]}. This downloads the public datasets."
        # The data steps need only Strands Decider's prompt code and datasets, so HMF's environment runs them, with
        # Strands Decider's source on the path
        (cd "$STRANDS_DIR" && PY="$PY" PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
            bash training/recipe.sh "${steps[@]}")
    else
        echo "Already built in $STRANDS_DIR/data."
    fi
    local link=data/strands-decider target
    target="$(cd "$STRANDS_DIR/data" && pwd)"
    if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
        echo "$link links to $target."
    elif [ -e "$link" ] || [ -L "$link" ]; then
        echo "$link exists and is not a link to $target. Move it away, or set STRANDS_DIR to its clone." >&2
        exit 1
    else
        ln -s "$target" "$link"
        echo "Linked $link to $target."
    fi
}

# strands_checksums FILE...: stops unless the files (paths within Strands Decider's data/ folder) match the checksums
# that Strands Decider recorded in data/SHA256SUMS
strands_checksums() {
    local sha256="sha256sum" name line lines=""
    command -v sha256sum >/dev/null || sha256="shasum -a 256"
    for name; do
        line="$(awk -v f="data/$name" '$2 == f' "$STRANDS_DIR/data/SHA256SUMS")"
        [ -n "$line" ] || { echo "Strands Decider records no checksum for data/$name." >&2; exit 1; }
        lines+="$line"$'\n'
    done
    (cd "$STRANDS_DIR" && printf '%s' "$lines" | $sha256 -c -) ||
        { echo "These files do not match Strands Decider's data/SHA256SUMS: $*" >&2; exit 1; }
}
