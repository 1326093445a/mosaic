#!/usr/bin/env bash
# Render docs/P17_JN1.md to a self-contained HTML page and an A4 PDF.
#
# Needs pandoc, and Chrome only for the PDF: there is no LaTeX in this
# environment, so `pandoc -o x.pdf` cannot work and the PDF is printed from
# the HTML instead. Without Chrome the HTML is still produced, and a browser's
# own "Print to PDF" on it gives the same result -- docs/report.css carries the
# print rules (A4, a page break before each section, no split tables).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SRC=docs/P17_JN1.md
OUT=docs/build
mkdir -p "$OUT"

TITLE="P17 / JN1 — OpenDDE gradient probe: project record"
LAST=$(grep -o '^## [0-9]\+\.' "$SRC" | tail -1 | tr -d '#. ')
SUB="Sections 1–${LAST} · generated $(date +%Y-%m-%d)"

pandoc "$SRC" \
    --standalone --self-contained \
    --resource-path=docs:docs/figures:. \
    --toc --toc-depth=2 \
    --metadata title="$TITLE" \
    --metadata subtitle="$SUB" \
    --css docs/report.css \
    --from gfm --to html5 \
    -o "$OUT/P17_JN1.html"
echo "wrote $OUT/P17_JN1.html"

CHROME=$(command -v google-chrome || command -v google-chrome-stable \
         || command -v chromium || command -v chromium-browser || true)
if [[ -z "$CHROME" ]]; then
    echo "no Chrome found; open the HTML and print it to PDF instead" >&2
    exit 0
fi

# A throwaway profile: Chrome will not print with a profile it cannot lock,
# and reusing the real one would contend with a running browser.
PROFILE=$(mktemp -d)
trap 'rm -rf "$PROFILE"' EXIT
"$CHROME" --headless --disable-gpu --no-sandbox \
    --user-data-dir="$PROFILE" --no-pdf-header-footer \
    --print-to-pdf="$PWD/$OUT/P17_JN1.pdf" \
    "file://$PWD/$OUT/P17_JN1.html" 2>/dev/null
echo "wrote $OUT/P17_JN1.pdf ($(pdfinfo "$OUT/P17_JN1.pdf" 2>/dev/null \
      | awk '/^Pages/{print $2" pages"}'))"

# The talk version and the mechanism animation are built separately, because
# they need matplotlib and ffmpeg where this script needs only pandoc:
#   python docs/make_mechanism_video.py --stills   # mp4, gif, one PNG per scene
#   python docs/make_presentation.py               # 12-slide deck, HTML + 16:9 PDF
