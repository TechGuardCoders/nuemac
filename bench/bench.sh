#!/bin/bash
# bench.sh: speed + sanity benchmark for a Nuemac server (any llama-server compatible API). No Python, only what ships with macOS.
#   nuemac bench                  decode x5, prefill at ~4K and ~16K tokens, 12 known-answer questions
#   nuemac bench --depth 32768    also: a needle in a ~32K-token prompt, decode speed at that depth
#   nuemac bench --quick          decode x3 + quality check only
#   nuemac qa                     quality check only
# Speeds are the server's own timings (llama.cpp "timings"), so client and HTTP overhead are excluded.
set -euo pipefail

PREFIX="${NUEMAC_PREFIX:-$HOME/.nuemac}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8080}"; HOST="${HOST:-127.0.0.1}"
REPS=5; DEPTH=0; QA=1; SPEED=1; PREFILL="4096 16384"
KEY="$(cat "$PREFIX/active" 2>/dev/null || echo inkling-2-mini)"
while [ $# -gt 0 ]; do
  case "$1" in
    --model) KEY="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --reps) REPS="$2"; shift ;;
    --depth) DEPTH="$2"; shift ;;
    --prefill) PREFILL="$2"; shift ;;
    --quick) REPS=3; PREFILL="" ;;
    --qa-only) SPEED=0 ;;
    --no-qa) QA=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
# shellcheck disable=SC1090
. "$SRC/models/$KEY.conf"
URL="http://$HOST:$PORT"
curl -sf "$URL/health" >/dev/null || { echo "no server on $URL (nuemac start)" >&2; exit 1; }
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT

get() { plutil -extract "$2" raw -o - "$1" 2>/dev/null || true; }
post() { curl -sS --max-time 10800 -H 'Content-Type: application/json' --data-binary @"$2" "$URL$1" -o "$3"; }
median() { sort -n | awk '{a[NR]=$1} END{if(NR==0){print "";exit} m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; printf "%.2f\n", m}'; }
r1() { awk -v x="$1" 'BEGIN{ if (x=="") print "-"; else printf "%.1f\n", x }'; }
# Deterministic filler prose: n words, seed s (a new seed per request, so nothing is cached)
words() {
  awk -v n="$1" -v s="$2" 'BEGIN{ srand(s);
    m = split("the a of and to in that is was for on with as by at from it an be this are or which one had not but have all were when we there can more if out so said what up its about into than them only other new some could time these two may then do first any my now such like our over man me even most made after also did many before must through back years where much your way well down should because each just those people how too little state good very make world still own see men work long get here between both life being under never day same another know while last might us great old year off come since against go came right used take three himself few house use during without again place around however home small found thought went say part once general high upon school every don does got united left number course war until always away something fact though water less public put think almost hand enough far took head yet government system better set told nothing night end why called didn eyes find going look asked later knew point next program city business give group toward young days let room president side social given present several order national possible rather second face per among form important often things looked early white case john become large big need four within felt along children saw best church ever least power development light thing seemed family interest want members mind country area others done turned although open god service certain kind problem began different door thus help sense means whole matter perhaps itself york times law human line above name example action company hands local show five history whether gave either act feet across taken past quite anything seen having death experience body word half really week free field car words order", w, " ");
    for (i = 1; i <= n; i++) printf "%s%s", w[int(rand() * m) + 1], (i % 14 == 0 ? ". " : " ") }'
}
complete_body() { # prompt-file n_predict -> body on stdout
  printf '{"prompt":"%s","n_predict":%s,"temperature":0.7,"seed":7,"ignore_eos":true,"cache_prompt":false}' "$(cat "$1")" "$2"
}
chat_body() { # content-file max_tokens
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0,"cache_prompt":false,"chat_template_kwargs":%s}' \
    "$MODEL_ID" "$(cat "$1")" "$2" "$QA_KWARGS"
}

CHIP="$(sysctl -n machdep.cpu.brand_string)"
GPU_CORES="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}')"
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
OS="$(sw_vers -productVersion)"
echo "== $MODEL_LABEL on $CHIP (${GPU_CORES:-?}-core GPU, $RAM_GB GB, macOS $OS)"

DEC=""; DEC_ALL=""; PF=""; MEMFREE=""; QA_SCORE=""; DEPTH_JSON=""
if [ "$SPEED" = 1 ]; then
  # Calibrate words -> tokens with a ~1,000-word prompt (also warms the server)
  words 1000 1 > "$T/p"; complete_body "$T/p" 8 > "$T/b"; post /completion "$T/b" "$T/r"
  TPW="$(awk -v n="$(get "$T/r" timings.prompt_n)" 'BEGIN{printf "%.4f", n/1000}')"

  printf 'Write a long, detailed essay about the history of maps and navigation.\\n\\n' > "$T/p"
  complete_body "$T/p" 256 > "$T/b"
  for i in $(seq "$REPS"); do
    post /completion "$T/b" "$T/r"
    v="$(get "$T/r" timings.predicted_per_second)"; DEC_ALL="$DEC_ALL $(r1 "$v")"
    echo "$v" >> "$T/dec"; printf "decode run %s: %.1f tok/s\n" "$i" "$v"
  done
  DEC="$(median < "$T/dec")"
  MEMFREE="$(memory_pressure 2>/dev/null | awk '/free percentage/{print $NF}')"

  for tgt in $PREFILL; do
    n=$(awk -v t="$tgt" -v r="$TPW" 'BEGIN{printf "%d", t/r}')
    words "$n" "$tgt" > "$T/p"; complete_body "$T/p" 1 > "$T/b"; post /completion "$T/b" "$T/r"
    pn="$(get "$T/r" timings.prompt_n)"; ps="$(get "$T/r" timings.prompt_per_second)"
    printf "prefill %s tokens: %.1f tok/s (%.1f s)\n" "$pn" "$ps" "$(awk -v a="$pn" -v b="$ps" 'BEGIN{print a/b}')"
    PF="$PF{\"tokens\":$pn,\"tok_s\":$(r1 "$ps")},"
  done

  if [ "$DEPTH" -gt 0 ]; then
    n=$(awk -v t="$DEPTH" -v r="$TPW" 'BEGIN{printf "%d", t/r}')
    { words $((n / 2)) 11; printf ' The secret code for the vault is 4417. '; words $((n / 2)) 12
      printf '\\n\\nWhat is the secret code for the vault mentioned above? Answer with the number only.'; } > "$T/p"
    chat_body "$T/p" 300 > "$T/b"
    echo "depth test: ~$DEPTH tokens, this takes a while..."
    post /v1/chat/completions "$T/b" "$T/r"
    pn="$(get "$T/r" timings.prompt_n)"; ps="$(get "$T/r" timings.prompt_per_second)"
    pm="$(get "$T/r" timings.prompt_ms)"; ds="$(get "$T/r" timings.predicted_per_second)"
    ans="$(get "$T/r" choices.0.message.content)"
    case "$ans" in *4417*) needle=pass ;; *) needle=fail ;; esac
    printf "depth %s tokens: prefill %.1f tok/s, first token after %.0f s, decode %.1f tok/s, needle %s\n" "$pn" "$ps" "$(awk -v m="$pm" 'BEGIN{print m/1000}')" "$ds" "$needle"
    DEPTH_JSON="\"depth\":{\"tokens\":$pn,\"prefill_tok_s\":$(r1 "$ps"),\"ttft_s\":$(awk -v m="$pm" 'BEGIN{printf "%.0f", m/1000}'),\"decode_tok_s\":$(r1 "$ds"),\"needle\":\"$needle\"},"
  fi
fi

if [ "$QA" = 1 ]; then
  ok=0; n=0
  while IFS='|' read -r q a; do
    n=$((n + 1)); printf '%s' "$q" > "$T/p"; chat_body "$T/p" 700 > "$T/b"
    post /v1/chat/completions "$T/b" "$T/r"
    c="$(get "$T/r" choices.0.message.content | tr '\n' ' ')"
    if printf '%s' "$c" | tr '[:upper:]' '[:lower:]' | grep -qF "$a"; then ok=$((ok + 1)); s=PASS; else s=FAIL; fi
    printf "%s  %-60.60s -> %.50s\n" "$s" "$q" "$c"
  done < "$SRC/bench/questions.txt"
  QA_SCORE="$ok/$n"; echo "quality check: $QA_SCORE"
fi

mkdir -p "$PREFIX/results"
OUT="$PREFIX/results/$(date +%Y%m%d-%H%M%S)-$KEY.json"
cat > "$OUT" <<EOF
{"model":"$MODEL_KEY","label":"$MODEL_LABEL","file":"${MODEL_FILE:-${MODEL_REPO:-}@${MODEL_REV:-}}","engine":"nuemac $(grep -m1 "^__version__" "$SRC/engine/server.py" 2>/dev/null | cut -d\" -f2) ${FAMILY:-} on oMLX ${OMLX_VERSION:-}","args":"${SERVER_ARGS:-ctx ${CTX:-}}",
 "machine":{"chip":"$CHIP","gpu_cores":"${GPU_CORES:-}","ram_gb":$RAM_GB,"macos":"$OS"},
 "decode_tok_s":${DEC:-null},"decode_runs":"${DEC_ALL# }","prefill":[${PF%,}],$DEPTH_JSON
 "qa":"$QA_SCORE","mem_free_pct":"${MEMFREE:-}","date":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
echo
[ -n "$DEC" ] && echo "decode (median of $REPS): $DEC tok/s    system memory free: ${MEMFREE:-?}"
echo "saved $OUT"
