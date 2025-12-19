# Session report generator

This script scans the repository `sessions/` directory and generates one HTML report per session that contains the required documents and the required text artifacts.

Usage examples

Run against the repository sessions directory and write reports to `reports/session_reports`:

```
python -m iris_demo.scripts.generate_session_reports --sessions-dir ./sessions --output-dir ./reports/session_reports
```

Do a dry-run (show what would be generated):

```
python -m iris_demo.scripts.generate_session_reports --sessions-dir ./sessions --output-dir ./reports/session_reports --dry-run
```

Configuration

- `--min-text-artifacts`: minimum number of text-like files (default: 3). The script also requires the three specific text artifacts described below.
- `--max-file-display-size`: truncation threshold for very large files (default: 200000 bytes)

Required text artifacts

Each qualifying session **must** include these files:

- `reconstruction.txt`
- `social_story.txt`
- `slp_report.txt`

File ordering in reports

Reports will order files so that `scenario.json` appears first, `metadata.json` is second, followed by any files beginning with `layer1` (e.g., `layer1_raw.json`), and then the rest alphabetically.

Appearance improvements

- JSON files are now syntax highlighted for easier reading.
- Long text files are truncated visually with a `Show more` / `Show less` toggle so they don't overwhelm the window. You can expand any file's content to view it fully.

Index page

An `index.html` file is generated in the output directory and lists all generated reports. Each row includes:

- **Session ID** (link to the session report)
- **Scenario** (the `name` field from `scenario.json`)
- **Layer2 model** (from `metadata.json` models: `layer2_llm_model`)
- **Reconstruction model** (from `metadata.json` models: `reconstruction_llm_model`)
- **Language** (from `metadata.json` `language` field)

If any values are missing, the index will show `(unknown)` for that column.

Report output

Each report is an HTML file named `<session_id>.html` containing a simple listing of files in the session folder. Files are shown inside collapsible `<details>` blocks and JSON files are pretty-printed.

Notes

The script detects "text" files by extension (`.txt`, `.md`, `.log`), by MIME-type guess, or by trying to decode the start of the file as UTF-8.
