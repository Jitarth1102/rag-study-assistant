import json
import pytest

from rag_assistant import cli


def test_cli_ingest_single_subject(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: "cfg")
    monkeypatch.setattr(cli.subject_service, "get_subject", lambda sid: {"subject_id": sid})

    calls = {}

    def fake_process(subject_id, config=None, force=False, limit=None):
        calls["args"] = (subject_id, config, force, limit)
        return {"indexed": 1, "failed": 0, "skipped_missing": 0, "details": [], "processed": 1}

    monkeypatch.setattr(cli.pipeline, "process_subject_new_assets", fake_process)

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "--subject", "math"])
    args.func(args)

    out = json.loads(capsys.readouterr().out)
    assert out["subjects"][0]["subject_id"] == "math"
    assert out["subjects"][0]["assets_indexed"] == 1
    assert calls["args"] == ("math", "cfg", False, None)


def test_cli_ingest_all_subjects_respects_limit(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: "cfg2")
    monkeypatch.setattr(cli.subject_service, "list_subjects", lambda: [{"subject_id": "s1"}, {"subject_id": "s2"}])
    monkeypatch.setattr(cli.subject_service, "get_subject", lambda sid: {"subject_id": sid})

    calls = []

    def fake_process(subject_id, config=None, force=False, limit=None):
        calls.append((subject_id, force, limit))
        return {"indexed": 1, "failed": 0, "skipped_missing": 0, "details": [], "processed": 1}

    monkeypatch.setattr(cli.pipeline, "process_subject_new_assets", fake_process)

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "--all-subjects", "--limit", "1", "--force"])
    args.func(args)

    out = json.loads(capsys.readouterr().out)
    assert len(out["subjects"]) == 1  # stopped after limit
    assert len(calls) == 1
    assert calls[0] == ("s1", True, 1)


def test_cli_ingest_missing_subject_exits(monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda: "cfg3")
    monkeypatch.setattr(cli.subject_service, "get_subject", lambda sid: None)

    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "--subject", "missing"])

    with pytest.raises(SystemExit) as excinfo:
        args.func(args)
    assert excinfo.value.code == 1
