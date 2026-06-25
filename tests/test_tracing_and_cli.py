import json

from conclave.cli import main
from conclave.server.dashboard import render_dashboard
from conclave.tracing import EventType, Tracer


def test_tracer_records_and_serializes():
    t = Tracer()
    t.record(EventType.NOTE, note="hello")
    with t.span("work"):
        t.record(EventType.STEP, index=1)
    summary = t.summary()
    assert summary["note"] == 1
    assert summary["step"] == 1
    data = json.loads(t.to_json())
    assert data["event_count"] == len(t.events)


def test_dashboard_renders_trace():
    t = Tracer()
    t.record(EventType.LLM_CALL, agent="a", text="hi")
    html = render_dashboard(t.to_dict())
    assert "<!DOCTYPE html>" in html
    assert "llm_call" in html


def test_cli_tools_command(capsys):
    rc = main(["tools"])
    assert rc == 0
    assert "calculator" in capsys.readouterr().out


def test_cli_run_single(capsys):
    rc = main(["run", "What is 8 * 8?", "--mode", "single"])
    assert rc == 0
    assert "64" in capsys.readouterr().out


def test_cli_run_orchestrate_with_trace(tmp_path):
    trace = tmp_path / "trace.json"
    rc = main(["run", "Explain unit testing", "--mode", "orchestrate", "--trace", str(trace)])
    assert rc == 0
    assert trace.exists()
    data = json.loads(trace.read_text())
    assert data["event_count"] > 0
