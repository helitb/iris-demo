#!/usr/bin/env python3
"""Generate HTML reports for sessions that contain a full set of docs.

Usage:
    python -m iris_demo.scripts.generate_session_reports --sessions-dir ../sessions --output-dir ./reports/session_reports

The script looks for sessions that contain the required files and at least a configurable
number of text artifacts (default: 3). For each qualifying session it creates a single
HTML file that lists all files in the session folder, showing their contents inside
collapsible blocks (<details>/<summary>).
"""
from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
from pathlib import Path
from typing import Iterable, List
import shutil


REQUIRED_FILES = {
    "layer1_raw.json",
    "layer2_raw.json",
    "layer2_sanitized.json",
    "metadata.json",
    "scenario.json",
}

# The required textual artifacts per session
REQUIRED_TEXT_ARTIFACTS = {"reconstruction.txt", "social_story.txt", "slp_report.txt"}

TEXT_EXTS = {".txt", ".md", ".log"}


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTS:
        return True
    mt, _ = mimetypes.guess_type(str(path))
    if mt is not None and mt.startswith("text"):
        return True
    # Fallback: try reading and decoding a small slice
    try:
        with path.open("rb") as f:
            chunk = f.read(2048)
            chunk.decode("utf-8")
        return True
    except Exception:
        return False


def find_sessions_with_full_docs(sessions_dir: Path, min_text_artifacts: int = 3) -> List[Path]:
    sessions = []
    for entry in sessions_dir.iterdir():
        if not entry.is_dir():
            continue
        names = {p.name for p in entry.iterdir() if p.is_file()}
        if not REQUIRED_FILES.issubset(names):
            continue
        # Require that the specific text artifacts exist
        if not REQUIRED_TEXT_ARTIFACTS.issubset(names):
            continue
        # Count text artifact files (any file with text-like extension or decodable)
        text_files = [p for p in entry.iterdir() if p.is_file() and is_text_file(p)]
        if len(text_files) < min_text_artifacts:
            continue
        sessions.append(entry)
    return sessions


def clean_incomplete_sessions(sessions_dir: Path, min_text_artifacts: int = 3) -> List[Path]:
    """Delete session directories that do not satisfy required files and text artifacts.

    Returns a list of deleted session paths.
    """
    deleted = []
    for entry in sessions_dir.iterdir():
        if not entry.is_dir():
            continue
        names = {p.name for p in entry.iterdir() if p.is_file()}
        if not REQUIRED_FILES.issubset(names) or not REQUIRED_TEXT_ARTIFACTS.issubset(names):
            try:
                shutil.rmtree(entry)
                deleted.append(entry)
            except Exception as exc:
                print(f"Failed to remove {entry}: {exc}")
    return deleted


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Session report: {session_id}</title>
  <style>
        body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; padding: 1rem; }}
        .file-meta {{ color: #666; font-size: .9rem; margin-bottom: .25rem; }}
        .file-block {{ margin-bottom: 1rem; }}
        pre.file-content {{ background: #f8f8f8; padding: .75rem; overflow: auto; border-radius: 6px; white-space: pre-wrap; word-break: break-word; }}
        .file-content.collapsed {{ max-height: 320px; }}
        details {{ margin-bottom: 1rem; border: 1px solid #eee; padding: .5rem; border-radius: 6px; }}
        .controls {{ margin-top: .5rem; margin-bottom: .5rem; }}
        button.toggle-btn {{ background: #007acc; color: white; border: none; padding: .35rem .6rem; border-radius: 4px; cursor: pointer; font-size: .9rem; }}
        /* simple JSON syntax highlighting */
        .json-key {{ color: #a71d5d; }}
        .json-string {{ color: #183691; }}
        .json-number {{ color: #0086b3; }}
        .json-boolean {{ color: #795da3; }}
        .json-null {{ color: #b52a1d; }}
  </style>
</head>
<body>
  <h1>Session report: {session_id}</h1>
  <p>Path: {session_path}</p>
  <hr/>
    {file_blocks}

    <script>
        // Toggle collapsed content for a file block
        function toggleContent(ev, id) {{
            var el = document.getElementById(id);
            if (!el) return;
            if (el.classList.contains('collapsed')) {{
                el.classList.remove('collapsed');
                ev.target.textContent = 'Show less';
            }} else {{
                el.classList.add('collapsed');
                ev.target.textContent = 'Show more';
            }}
        }}

        // Simple JSON syntax highlighter applied to elements with data-json
        document.addEventListener('DOMContentLoaded', function() {{
            var nodes = document.querySelectorAll('pre[data-json]');
            nodes.forEach(function(node) {{
                var text = node.textContent;
                // escape already present HTML is fine because we operate on textContent
                // highlight keys
                var highlighted = text
                    .replace(/(\"(.*?)\")\s*:/g, '<span class="json-key">$1</span>:')
                    .replace(/:\s*\"(.*?)\"/g, ': <span class="json-string">"$1"</span>')
                    .replace(/\b(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/g, '<span class="json-number">$1</span>')
                    .replace(/\b(true|false)\b/g, '<span class="json-boolean">$1</span>')
                    .replace(/\b(null)\b/g, '<span class="json-null">$1</span>');
                node.innerHTML = highlighted;
            }});
        }});
    </script>
</body>
</html>
"""


def make_file_block(path: Path, max_display_size: int = 200_000) -> str:
    size = path.stat().st_size
    name = html.escape(path.name)
    meta = f"<div class=\"file-meta\">Size: {size} bytes</div>"

    # Try to render JSON prettily
    try:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
                content = json.dumps(obj, indent=2, ensure_ascii=False)
                content = html.escape(content)
                # Use data-json attribute for JS highlighter and collapsed class by default
                el_id = f"file-{abs(hash(str(path)))}"
                return (f"<details class=\"file-block\"><summary><strong>{name}</strong></summary>"
                    f"{meta}<div class=\"controls\"><button class=\"toggle-btn\" onclick=\"toggleContent(event, '{el_id}')\">Show more</button></div>"
                    f"<pre id=\"{el_id}\" class=\"file-content collapsed\" data-json>{content}</pre></details>")
    except Exception:
        # fall through and try to treat as text
        pass

    # If file is too big, show hint and a truncated preview
    if size > max_display_size:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                preview = f.read(4096)
        except Exception:
            preview = "(unable to preview)"
        preview = html.escape(preview)
        hint = f"<div class=\"file-meta\">(truncated preview, file > {max_display_size} bytes)</div>"
        return f"<details><summary><strong>{name}</strong></summary>{meta}{hint}<pre>{preview}</pre></details>"

    # Attempt to read as text
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        content = html.escape(content)
        el_id = f"file-{abs(hash(str(path)))}"
        return (f"<details class=\"file-block\"><summary><strong>{name}</strong></summary>"
                f"{meta}<div class=\"controls\"><button class=\"toggle-btn\" onclick=\"toggleContent(event, '{el_id}')\">Show more</button></div>"
                f"<pre id=\"{el_id}\" class=\"file-content collapsed\">{content}</pre></details>")
    except Exception:
        return f"<details><summary><strong>{name}</strong></summary>{meta}<pre>(binary or unreadable file)</pre></details>"


def generate_report_for_session(session_path: Path, out_path: Path, max_display_size: int = 200_000) -> None:
    session_id = session_path.name
    # Ordering rule: show `scenario.json` first, then `metadata.json`, then any l1 reports (starting with 'layer1'),
    # then the remaining files alphabetically. This ensures metadata appears second under scenario and before l1 reports.
    all_files = [p for p in session_path.iterdir() if p.is_file()]
    def order_key(p: Path):
        name = p.name
        if name == "scenario.json":
            return (0, name)
        if name == "metadata.json":
            return (1, name)
        if name.startswith("layer1"):
            return (2, name)
        return (3, name)
    files = sorted(all_files, key=order_key)
    blocks = [make_file_block(p, max_display_size=max_display_size) for p in files]
    html_text = HTML_TEMPLATE.format(session_id=html.escape(session_id), session_path=html.escape(str(session_path)), file_blocks="\n".join(blocks))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(html_text)


def generate_index(sessions: List[Path], out_dir: Path) -> None:
    """Generate an index.html in out_dir listing sessions with scenario name and models."""
    rows = []
    for s in sessions:
        sid = s.name
        scenario_name = "(unknown)"
        layer2_model = "(unknown)"
        recon_model = "(unknown)"
        language = "(unknown)"
        try:
            scen = json.loads((s / "scenario.json").read_text(encoding="utf-8"))
            scenario_name = scen.get("name") or scen.get("scenario") or scenario_name
        except Exception:
            pass
        try:
            meta = json.loads((s / "metadata.json").read_text(encoding="utf-8"))
            models = meta.get("models", {}) or {}
            layer2_model = models.get("layer2_llm_model") or models.get("layer2") or layer2_model
            recon_model = models.get("reconstruction_llm_model") or models.get("reconstruction") or recon_model
            language = meta.get("language") or language
        except Exception:
            pass
        report_path = out_dir / f"{sid}.html"
        rows.append((sid, scenario_name, layer2_model, recon_model, language, report_path.name))

    index_lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Sessions index</title></head>",
        "<body>",
        "<h1>Session reports index</h1>",
        "<table border=\"1\" cellpadding=\"6\" style=\"border-collapse:collapse\">",
        "<thead><tr><th>Session ID</th><th>Scenario</th><th>Layer2 model</th><th>Reconstruction model</th><th>Language</th></tr></thead>",
        "<tbody>",
    ]
    for sid, scenario_name, layer2_model, recon_model, language, fname in rows:
        index_lines.append(f"<tr><td><a href=\"{fname}\">{sid}</a></td><td>{html.escape(str(scenario_name))}</td><td>{html.escape(str(layer2_model))}</td><td>{html.escape(str(recon_model))}</td><td>{html.escape(str(language))}</td></tr>")
    index_lines.extend(["</tbody>", "</table>", "</body>", "</html>"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate HTML session reports")
    parser.add_argument("--sessions-dir", default="./sessions", help="Path to sessions directory")
    parser.add_argument("--output-dir", default="./reports/session_reports", help="Output directory for reports")
    parser.add_argument("--min-text-artifacts", type=int, default=3, help="Minimum number of text files required to consider a session complete")
    parser.add_argument("--max-file-display-size", type=int, default=200_000, help="Max file size (bytes) to display fully in the report; larger files are truncated")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without writing files")
    parser.add_argument("--clean-missing", action="store_true", help="Delete session folders missing required artifacts before generating reports")
    parser.add_argument("--include-all", action="store_true", help="Generate reports for all session folders (not only those with complete artifacts)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    sessions_dir = Path(args.sessions_dir)
    out_dir = Path(args.output_dir)

    if not sessions_dir.exists():
        print(f"Sessions directory does not exist: {sessions_dir}")
        return 2

    if args.clean_missing:
        deleted = clean_incomplete_sessions(sessions_dir, min_text_artifacts=args.min_text_artifacts)
        print(f"Removed {len(deleted)} incomplete session(s).")
        # After cleaning, include all remaining sessions
        args.include_all = True

    if args.include_all:
        matches = [p for p in sessions_dir.iterdir() if p.is_dir()]
    else:
        matches = find_sessions_with_full_docs(sessions_dir, min_text_artifacts=args.min_text_artifacts)
    if not matches:
        print("No sessions matching required files + text artifact count found.")
        return 0

    print(f"Found {len(matches)} session(s) to report on.")
    # Remove previous reports directory to ensure fresh output
    try:
        if out_dir.exists():
            # Remove files contained inside out_dir
            for p in out_dir.iterdir():
                if p.is_file():
                    p.unlink()
                else:
                    # For directories, remove files recursively
                    for sub in p.rglob("*"):
                        if sub.is_file():
                            sub.unlink()
                    try:
                        p.rmdir()
                    except Exception:
                        pass
    except Exception as exc:
        print(f"Warning: failed to clean output dir {out_dir}: {exc}")

    for s in matches:
        out_file = out_dir / f"{s.name}.html"
        print(f"Writing report for {s.name} -> {out_file}")
        if not args.dry_run:
            try:
                generate_report_for_session(s, out_file, max_display_size=args.max_file_display_size)
            except Exception as exc:
                print(f"Failed to generate report for {s}: {exc}")
    # Generate index page linking to generated reports
    try:
        generate_index(matches, out_dir)
        print(f"Wrote index -> {out_dir / 'index.html'}")
    except Exception as exc:
        print(f"Failed to generate index: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
