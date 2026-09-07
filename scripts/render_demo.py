"""Render docs/demo/session.json as an animated terminal SVG.

The transcript is JSON, the output is one self-contained SVG whose animation is
plain CSS, and GitHub renders it inline; no recorder, converter or font is
needed to regenerate it.

    python scripts/render_demo.py                   # docs/demo/tactistat-demo.svg
    python scripts/render_demo.py --static out.svg  # every scene at once, for checking
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "docs" / "demo" / "session.json"
OUTPUT = ROOT / "docs" / "demo" / "tactistat-demo.svg"

COLS, ROWS = 100, 26
FONT_PX, CHAR_W, LINE_H = 13, 7.85, 20
PAD, HEADER = 18, 34
# Pacing, in milliseconds.
TYPE_MS, ENTER_MS, LINE_MS, GAP_MS = 36, 380, 24, 400
HOLD_MS, HOLD_PER_CHAR_MS, MAX_HOLD_MS = 2800, 9, 7000

BG, CHROME, TITLE = "#1a1a19", "#2c2c2a", "#c3c2b7"
TEXT, DIM, PROMPT, WARN, ROUTE = "#e8e6df", "#9a9891", "#3987e5", "#eb6834", "#1baf7a"
FONT = '"SF Mono", Menlo, Consolas, "DejaVu Sans Mono", "Liberation Mono", monospace'
ROUTES = ("STAT", "TACTICAL", "HYBRID")


def wrap(line: str, width: int = COLS) -> list[str]:
    """Wrap at spaces, keeping the original indentation on every piece."""
    indent = len(line) - len(line.lstrip(" "))
    pieces, rest = [], line
    while len(rest) > width:
        cut = rest.rfind(" ", indent + 1, width)
        if cut <= indent:
            cut = width
        pieces.append(rest[:cut].rstrip())
        rest = " " * indent + rest[cut:].lstrip()
    pieces.append(rest)
    return pieces


def colour(line: str) -> tuple[str, str | None]:
    """The fill for an output line, and a second fill for its tail if it has one."""
    stripped = line.strip()
    if stripped.startswith("!"):
        return WARN, None
    if stripped.startswith(("[", "{", "Sources:", "matches for", "...")):
        return DIM, None
    if line.startswith(ROUTES) and " " in line:
        return ROUTE, DIM
    if stripped.startswith(("STATS TOOL", "RAG TOOL", "Ask in")):
        return TITLE, None
    return TEXT, None


def tspan(text: str, fill: str, extra: str = "") -> str:
    return f'<tspan fill="{fill}"{extra}>{escape(text)}</tspan>'


class Renderer:
    def __init__(self, session: dict, static: bool):
        self.session, self.static = session, static
        self.keyframes: list[str] = []
        self.body: list[str] = []
        self.t = 900  # a beat before the first keystroke

    def key(self, name: str, on_ms: float, off_ms: float | None = None) -> None:
        """A keyframe that switches an element on at on_ms (and off at off_ms)."""
        self.keyframes.append((name, on_ms, off_ms))

    def render(self) -> str:
        scenes = self.session["scenes"]
        y_offset = 0
        for index, scene in enumerate(scenes):
            lines = self.scene_lines(scene, index)
            if len(lines) > ROWS:
                raise SystemExit(
                    f"scene {index + 1} needs {len(lines)} rows; the window has {ROWS}"
                )
            self.body.append(self.scene_svg(lines, index, y_offset))
            if self.static:
                y_offset += (len(lines) + 1) * LINE_H
        total = self.t + 600
        height = HEADER + 2 * PAD + (y_offset if self.static else ROWS * LINE_H)
        width = 2 * PAD + int(COLS * CHAR_W)
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}" font-family=\'{FONT}\' font-size="{FONT_PX}">',
                self.style(total),
                f'<rect width="{width}" height="{height}" rx="10" fill="{BG}"/>',
                f'<path d="M0 10a10 10 0 0 1 10-10h{width - 20}a10 10 0 0 1 10 10'
                f'v{HEADER - 10}H0z" fill="{CHROME}"/>',
                *(
                    f'<circle cx="{18 + 20 * i}" cy="{HEADER / 2}" r="5.5" fill="{c}"/>'
                    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840"))
                ),
                f'<text x="{width / 2}" y="{HEADER / 2 + 4.5}" text-anchor="middle" fill="{TITLE}" '
                f'font-size="12">{escape(self.session.get("title", ""))}</text>',
                *self.body,
                "</svg>",
            ]
        )

    def style(self, total: float) -> str:
        if self.static:
            return ""
        rules = [f".a{{animation:{total:.0f}ms linear infinite both}}"]
        for name, on_ms, off_ms in self.keyframes:
            on = 100 * on_ms / total
            stops = [f"0%,{on:.3f}%{{opacity:0}}"]
            if off_ms is None:
                stops.append(f"{on + 0.001:.3f}%,100%{{opacity:1}}")
            else:
                off = 100 * off_ms / total
                stops.append(f"{on + 0.001:.3f}%,{off:.3f}%{{opacity:1}}")
                stops.append(f"{off + 0.001:.3f}%,100%{{opacity:0}}")
            rules.append(f"@keyframes {name}{{{''.join(stops)}}}")
        return "<style>" + "".join(rules) + "</style>"

    def scene_lines(self, scene: dict, index: int) -> list[dict]:
        """Lay a scene out as timed lines; typing and reveals advance the clock."""
        lines: list[dict] = []
        chars = 0
        for step in scene["steps"]:
            prompt, typed = step.get("prompt", "$ "), step.get("input", "")
            for piece_index, piece in enumerate(wrap(prompt + typed)):
                spans = []
                if piece_index == 0:
                    spans.append({"text": prompt, "fill": PROMPT, "at": self.t})
                    piece = piece[len(prompt) :]
                for char in piece:
                    spans.append({"text": char, "fill": TEXT, "at": self.t})
                    self.t += TYPE_MS
                lines.append({"spans": spans, "at": lines[-1]["at"] if piece_index else self.t})
            self.t += ENTER_MS
            for raw in step.get("output", "").split("\n"):
                fill, tail = colour(raw)
                for piece_index, piece in enumerate(wrap(raw)):
                    if tail and piece_index == 0 and " " in piece:
                        head, rest = piece.split(" ", 1)
                        spans = [{"text": head, "fill": fill}, {"text": " " + rest, "fill": tail}]
                    else:
                        spans = [{"text": piece, "fill": tail if tail and piece_index else fill}]
                    lines.append({"spans": spans, "at": self.t})
                    chars += len(piece)
                    self.t += LINE_MS
        self.t += min(MAX_HOLD_MS, HOLD_MS + chars * HOLD_PER_CHAR_MS)
        return lines

    def scene_svg(self, lines: list[dict], index: int, y_offset: int) -> str:
        start = min(line["at"] for line in lines)
        end = self.t
        self.t += GAP_MS
        parts = []
        for row, line in enumerate(lines):
            y = HEADER + PAD + y_offset + (row + 1) * LINE_H - 5
            spans = []
            for column, span in enumerate(line["spans"]):
                extra = ""
                if not self.static and "at" in span:
                    name = f"c{index}_{row}_{column}"
                    self.key(name, span["at"])
                    extra = f' class="a" style="animation-name:{name}"'
                spans.append(tspan(span["text"], span["fill"], extra))
            attrs = ""
            if not self.static:
                name = f"l{index}_{row}"
                self.key(name, line["at"])
                attrs = f' class="a" style="animation-name:{name}"'
            parts.append(
                f'<text x="{PAD}" y="{y}" xml:space="preserve"{attrs}>{"".join(spans)}</text>'
            )
        group = ""
        if not self.static:
            self.key(f"s{index}", start - 1, end)
            group = f' class="a" style="animation-name:s{index}"'
        return f"<g{group}>" + "".join(parts) + "</g>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=str(SESSION))
    parser.add_argument("--static", metavar="PATH", help="write an unanimated, stacked version")
    args = parser.parse_args()
    session = json.loads(Path(args.session).read_text())
    output = Path(args.static) if args.static else OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(Renderer(session, static=bool(args.static)).render())
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
