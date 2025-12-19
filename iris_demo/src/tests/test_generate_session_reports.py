import json
import sys
from pathlib import Path

# Ensure package import works when tests are run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from iris_demo.scripts import generate_session_reports as gr


def create_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_find_and_generate(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    s1 = sessions_dir / "sess1"
    s1.mkdir(parents=True)

    # required files
    create_file(s1 / "layer1_raw.json", json.dumps({"a": 1}))
    create_file(s1 / "layer2_raw.json", json.dumps({"b": 2}))
    create_file(s1 / "layer2_sanitized.json", json.dumps({"c": 3}))
    create_file(s1 / "metadata.json", json.dumps({"meta": 1}))
    create_file(s1 / "scenario.json", json.dumps({"scenario": 1}))

    # text artifacts (the required three defined by the project)
    create_file(s1 / "reconstruction.txt", "reconstruction text")
    create_file(s1 / "social_story.txt", "social story text")
    create_file(s1 / "slp_report.txt", "slp report text")

    found = gr.find_sessions_with_full_docs(sessions_dir, min_text_artifacts=3)
    assert len(found) == 1 and found[0].name == "sess1"

    out_dir = tmp_path / "reports"
    out_file = out_dir / "sess1.html"
    gr.generate_report_for_session(found[0], out_file)
    assert out_file.exists()
    contents = out_file.read_text(encoding="utf-8")
    # Should include filenames and obey ordering (scenario then metadata then layer1)
    assert "reconstruction.txt" in contents
    assert "social_story.txt" in contents
    assert "slp_report.txt" in contents
    # check ordering: scenario appears before metadata, and metadata before layer1
    idx_scenario = contents.index("scenario.json")
    idx_metadata = contents.index("metadata.json")
    idx_layer1 = contents.index("layer1_raw.json")
    assert idx_scenario < idx_metadata < idx_layer1
    # Should include UI elements for toggling and JSON highlighting
    assert 'class="file-content collapsed"' in contents
    assert 'data-json' in contents
    assert 'toggleContent' in contents
    # Now generate an index and ensure it contains scenario name and models
    gr.generate_index(found, out_dir)
    index_file = out_dir / "index.html"
    assert index_file.exists()
    index_contents = index_file.read_text(encoding="utf-8")
    assert "sess1" in index_contents
    # Default test session has no models, so the placeholders may be present
    assert "(unknown)" in index_contents or "layer2" in index_contents
    # Language column header should be present and default language placeholder may appear
    assert "<th>Language</th>" in index_contents
    assert "(unknown)" in index_contents


def test_clean_incomplete_and_include_all(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    complete = sessions_dir / "complete"
    incomplete = sessions_dir / "incomplete"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)

    # create required files in complete
    for name in ["layer1_raw.json", "layer2_raw.json", "layer2_sanitized.json", "metadata.json", "scenario.json"]:
        create_file(complete / name, json.dumps({name: True}))
    for t in ["reconstruction.txt", "social_story.txt", "slp_report.txt"]:
        create_file(complete / t, "text")

    # incomplete only has scenario.json
    create_file(incomplete / "scenario.json", json.dumps({"name": "incomplete"}))

    # clean
    deleted = gr.clean_incomplete_sessions(sessions_dir)
    assert any(d.name == "incomplete" for d in deleted)
    assert (incomplete).exists() is False

    # generate reports with include_all (should include 'complete')
    out_dir = tmp_path / "reports"
    # use include_all by directly invoking generate_report_for_session on remaining dirs
    remaining = [d for d in sessions_dir.iterdir() if d.is_dir()]
    assert [d.name for d in remaining] == ["complete"]
    gr.generate_report_for_session(remaining[0], out_dir / "complete.html")
    assert (out_dir / "complete.html").exists()
