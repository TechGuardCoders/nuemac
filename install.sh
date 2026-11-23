#!/bin/bash
# Nuemac installer: the Nuemac engine plus a supported model, as a local OpenAI-compatible server.
# Tuned on a base 16 GB M4 Mac mini.
#
#   curl -fsSL https://raw.githubusercontent.com/TechGuardCoders/nuemac/main/install.sh | bash
#
# Options:
#   --model NAME        model to install (default: inkling-2-mini on 16 GB Macs, inkling-2-mini-oq3e on 24 GB+;
#                       list: docs/supported-models.md)
#   --prefix DIR        install here (default ~/.nuemac)
#   --port N            server port (default 8080, loopback only)
#   --no-start          install only, don't start the server
#   --no-wired-limit    don't offer to raise the GPU memory limit
#   --no-path           don't add nuemac to your PATH
#   --dry-run           check the Mac and print the plan, change nothing
set -euo pipefail

REPO="TechGuardCoders/nuemac"
BRANCH="${NUEMAC_BRANCH:-main}"
PREFIX="${NUEMAC_PREFIX:-$HOME/.nuemac}"
MODEL=""; PORT=""; START=1; WIRED=1; ADDPATH=1; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model|--models) MODEL="$2"; shift ;;
    --prefix) PREFIX="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --no-start) START=0 ;;
    --no-wired-limit) WIRED=0 ;;
    --no-path) ADDPATH=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done

b() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ask() { # question -> 0 for yes; needs a terminal (works under curl | bash via /dev/tty)
  local a=""; [ -r /dev/tty ] || return 1
  printf '%s [y/N] ' "$1" > /dev/tty; read -r a < /dev/tty || return 1
  case "$a" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}
gib() { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1e9}'; }
omlx_python() { for p in /opt/homebrew/opt/omlx/libexec/bin/python /usr/local/opt/omlx/libexec/bin/python; do
  [ -x "$p" ] && { echo "$p"; return 0; }; done; return 1; }

# ---- 1. check the Mac -------------------------------------------------------------------------
b "Nuemac installer"
[ "$(uname -s)" = Darwin ] || die "Nuemac is for macOS"
[ "$(uname -m)" = arm64 ] || die "needs Apple Silicon (M1 or newer)"
OSMAJ="$(sw_vers -productVersion | cut -d. -f1)"
[ "$OSMAJ" -ge 14 ] || die "needs macOS 14 or newer (you have $(sw_vers -productVersion))"
MEM_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))
[ "$MEM_MB" -ge 15000 ] || die "needs 16 GB of memory or more"
echo "Mac        $(sysctl -n machdep.cpu.brand_string), $((MEM_MB / 1024)) GB, macOS $(sw_vers -productVersion)"
# Default model by memory: the 16 GB build (96K) or, with 24 GB or more, the 24 GB build (256K).
if [ -z "$MODEL" ]; then
  if [ "$MEM_MB" -ge 23000 ]; then MODEL="inkling-2-mini-oq3e"; else MODEL="inkling-2-mini"; fi
fi
BREW="$(command -v brew || { [ -x /opt/homebrew/bin/brew ] && echo /opt/homebrew/bin/brew; } || true)"
if [ -z "$BREW" ] && ! omlx_python >/dev/null; then
  die "Nuemac runs on oMLX's Python stack, installed with Homebrew. Install Homebrew first:
  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"
then run this installer again."
fi

# ---- 2. get Nuemac ----------------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$HERE" ] && [ -f "$HERE/engine/server.py" ] && [ -d "$HERE/models" ]; then
  SRC_IN="$HERE"
else
  SRC_IN="$(mktemp -d)/nuemac"; mkdir -p "$SRC_IN"
  curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/heads/$BRANCH" | tar xz -C "$SRC_IN" --strip-components 1 \
    || die "could not download Nuemac from github.com/$REPO"
fi
CONF="$SRC_IN/models/$MODEL.conf"
[ -f "$CONF" ] || die "unknown model '$MODEL' (supported: $(ls "$SRC_IN/models" | sed 's/\.conf$//' | tr '\n' ' '))"
# shellcheck disable=SC1090
. "$CONF"
DEST="$PREFIX/models/$MODEL_DIR"
[ "$MEM_MB" -ge "${MIN_MEM_MB:-15000}" ] || die "$MODEL needs a Mac with $(( ${MIN_MEM_MB:-15000} / 1000 + 1 )) GB of memory or more"

# An earlier install of this model by the inkling2-mini-mac-recipe (mini-llm): reuse its verified files.
OLD="$HOME/.mini-llm/models/$MODEL_DIR"
REUSE=0
if [ ! -f "$DEST/.verified" ] && [ -f "$OLD/.verified" ]; then REUSE=1; fi

have=0
[ -d "$DEST" ] && have=$(( $(du -sk "$DEST" | awk '{print $1}') * 1024 ))
NEED=0
if [ ! -f "$DEST/.verified" ] && [ "$REUSE" = 0 ]; then
  NEED=$(( (MODEL_BYTES - have) / 1000000 )); [ "$NEED" -gt 0 ] || NEED=0
fi
PLAN="$MODEL_LABEL [$(gib "$MODEL_BYTES")]"
[ "$REUSE" = 1 ] && PLAN="$PLAN, reusing the files from ~/.mini-llm"
if ! omlx_python >/dev/null; then NEED=$(( NEED + 2000 )); PLAN="$PLAN + oMLX from Homebrew [~2 GB]"; fi
mkdir -p "$PREFIX" 2>/dev/null || true
FREE_MB=$(( $(df -k "$PREFIX" | awk 'NR==2{print $4}') / 1024 ))
echo "Install to $PREFIX  (free disk $((FREE_MB / 1024)) GB, download ~$((NEED / 1000)) GB)"
echo "Model      $PLAN"
[ "$FREE_MB" -gt $(( NEED + 3000 )) ] || die "not enough free disk space"
if [ "$DRY" = 1 ]; then echo "dry run: nothing changed"; exit 0; fi

mkdir -p "$PREFIX"/{bin,models,logs,results}
if [ "$(cd "$SRC_IN" && pwd -P)" != "$(cd "$PREFIX" && pwd -P)/nuemac" ]; then
  rm -rf "$PREFIX/nuemac.new"; cp -R "$SRC_IN" "$PREFIX/nuemac.new"
  rm -rf "$PREFIX/nuemac"; mv "$PREFIX/nuemac.new" "$PREFIX/nuemac"
fi
install -m 755 "$PREFIX/nuemac/bin/nuemac" "$PREFIX/bin/nuemac"
if [ -n "$PORT" ]; then   # replace only the PORT line: other settings (MODEL_PATH, SERVER_ARGS) survive an update
  { grep -v '^PORT=' "$PREFIX/settings" 2>/dev/null || true; echo "PORT=$PORT"; } > "$PREFIX/settings.new"
  mv "$PREFIX/settings.new" "$PREFIX/settings"
fi
echo "$MODEL" > "$PREFIX/active"

verify() { # file sha256
  printf "  verifying %s ... " "$(basename "$1")"
  if [ "$(shasum -a 256 "$1" | awk '{print $1}')" = "$2" ]; then echo ok; return 0; fi
  echo "checksum mismatch"; return 1
}
fetch() { # url dest sha256
  local part="$2.part"
  if [ -f "$2" ]; then verify "$2" "$3" && return 0; mv "$2" "$part"; fi
  echo "  downloading $(basename "$2")"
  for _ in 1 2 3 4 5; do
    curl -fL --retry 5 --retry-delay 5 -C - --progress-bar -o "$part" "$1" && break
    echo "  download interrupted, resuming..."; sleep 5
  done
  verify "$part" "$3" || { rm -f "$part"; die "download of $(basename "$2") was corrupt; run the installer again"; }
  mv "$part" "$2"
}

# ---- 3. engine: oMLX's Python stack (MLX, mlx-lm, the model code) ------------------------------
b "Engine"
if ! omlx_python >/dev/null; then
  # oMLX's formula lives in github.com/jundot/omlx (not the default homebrew-omlx name), so tap it by URL first.
  echo "  installing oMLX from Homebrew (tap jundot/omlx from github.com/jundot/omlx)"
  "$BREW" tap jundot/omlx https://github.com/jundot/omlx || die "could not add the jundot/omlx Homebrew tap"
  "$BREW" install jundot/omlx/omlx || die "brew install jundot/omlx/omlx failed"
fi
PY="$(omlx_python)" || die "oMLX installed but its Python was not found"
HAVE="$("$PY" -I -c 'import importlib.metadata as m; print(m.version("omlx"))' 2>/dev/null || echo unknown)"
if case " $OMLX_VERSION " in *" $HAVE "*) true;; *) false;; esac; then echo "  oMLX $HAVE"
else echo "  oMLX $HAVE installed; $MODEL was tested with $OMLX_VERSION (Nuemac checks itself at start-up)"; fi
echo "  nuemac $(grep -m1 '^__version__' "$PREFIX/nuemac/engine/server.py" | cut -d'"' -f2)"

# ---- 4. model -----------------------------------------------------------------------------------
b "Model"
if [ -f "$DEST/.verified" ]; then
  echo "  $MODEL_DIR already installed"
else
  mkdir -p "$DEST"
  if [ "$REUSE" = 1 ]; then
    echo "  linking the files of the earlier install (~/.mini-llm), then checking each one"
    for f in $MODEL_FILES; do
      [ -f "$DEST/${f%%:*}" ] || ln "$OLD/${f%%:*}" "$DEST/${f%%:*}" 2>/dev/null || cp "$OLD/${f%%:*}" "$DEST/${f%%:*}"
    done
  fi
  for f in $MODEL_FILES; do
    fetch "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REV/${f%%:*}" "$DEST/${f%%:*}" "${f##*:}"
  done
  echo "$MODEL_REPO@$MODEL_REV" > "$DEST/.verified"
fi

# ---- 5. GPU memory limit ---------------------------------------------------------------------
# macOS lets the GPU wire only ~2/3 of memory by default (~10.7 GB on 16 GB, ~16 GB on 24 GB). Each model's conf
# says how much it needs (GPU_MB); the limit is raised to that plus a little slack.
WANT=$(( ${GPU_MB:-13700} + 800 ))
LIM="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
EFF="$LIM"; [ "$EFF" -gt 0 ] || EFF=$(( MEM_MB * 2 / 3 ))
if [ "$WIRED" = 1 ] && [ "$EFF" -lt "$WANT" ]; then
  b "GPU memory limit"
  echo "  The GPU may use only ~$EFF MB now; $MODEL needs $WANT MB."
  echo "  Raising it needs your password (sudo). Close other big apps while the model runs."
  if ask "  Raise it to $WANT MB now?"; then
    sudo sysctl iogpu.wired_limit_mb="$WANT" < /dev/tty
    if ask "  Keep it after reboot (installs a small boot job in /Library/LaunchDaemons)?"; then
      sed "s/iogpu.wired_limit_mb=[0-9]*/iogpu.wired_limit_mb=$WANT/" "$PREFIX/nuemac/support/com.nuemac.wiredlimit.plist" |
        sudo tee /Library/LaunchDaemons/com.nuemac.wiredlimit.plist > /dev/null
      sudo launchctl bootstrap system /Library/LaunchDaemons/com.nuemac.wiredlimit.plist 2>/dev/null || true
    fi
  else
    echo "  Skipped. Before starting, run:  sudo sysctl iogpu.wired_limit_mb=$WANT"
  fi
fi

# ---- 6. PATH ------------------------------------------------------------------------------------
if [ "$ADDPATH" = 1 ]; then
  mkdir -p "$HOME/.local/bin"; ln -sf "$PREFIX/bin/nuemac" "$HOME/.local/bin/nuemac"
  if ! grep -qs '.local/bin' "$HOME/.zprofile"; then
    printf '\n# nuemac\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$HOME/.zprofile"
  fi
fi

# ---- 7. start -------------------------------------------------------------------------------------
if [ "$START" = 1 ]; then
  # The earlier recipe's login service would hold the same port: stop it (its files stay).
  if launchctl print "gui/$(id -u)/com.mini-llm.server" >/dev/null 2>&1; then
    echo "  stopping the earlier mini-llm login service (Nuemac replaces it; ~/.mini-llm is left in place)"
    launchctl bootout "gui/$(id -u)/com.mini-llm.server" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/com.mini-llm.server.plist"
    sleep 3
  fi
  b "Starting $MODEL"
  NUEMAC_PREFIX="$PREFIX" "$PREFIX/bin/nuemac" restart
  NUEMAC_PREFIX="$PREFIX" "$PREFIX/bin/nuemac" qa | grep "quality check"
fi

b "Done"
cat <<EOF
  nuemac status            what's installed and running
  nuemac models            supported models
  nuemac bench             measure speed + quality on your Mac (~10 min)
  nuemac stop | start      the server also starts by itself at login
  API (OpenAI-compatible): http://127.0.0.1:${PORT:-8080}/v1   model: $MODEL_ID
  Open a new terminal (or run: export PATH="\$HOME/.local/bin:\$PATH") to use nuemac.
EOF
