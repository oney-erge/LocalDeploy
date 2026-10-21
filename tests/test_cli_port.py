"""The pip/uvx entry point must not point the user at a port that another program already owns."""
from __future__ import annotations

from localdeploy import cli


def test_free_port_is_used_as_is(monkeypatch):
    monkeypatch.setattr(cli, "port_in_use", lambda host, port: False)

    assert cli.choose_port("127.0.0.1", 8000, explicit=False) == 8000


def test_default_port_falls_back_to_the_next_free_one(monkeypatch, capsys):
    monkeypatch.setattr(cli, "port_in_use", lambda host, port: port in {8000, 8001})

    assert cli.choose_port("127.0.0.1", 8000, explicit=False) == 8002
    assert "Port 8000 is already in use; using 8002" in capsys.readouterr().out


def test_explicit_busy_port_is_refused_with_a_suggestion(monkeypatch, capsys):
    monkeypatch.setattr(cli, "port_in_use", lambda host, port: True)

    assert cli.choose_port("127.0.0.1", 8000, explicit=True) is None
    err = capsys.readouterr().err
    assert "Port 8000 is already in use" in err
    assert "--port 8001" in err


def test_no_free_port_in_range_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(cli, "port_in_use", lambda host, port: True)

    assert cli.choose_port("127.0.0.1", 8000, explicit=False) is None
    assert "are all in use" in capsys.readouterr().err


def test_main_exits_nonzero_without_starting_a_server_when_the_port_is_taken(monkeypatch):
    started = []
    monkeypatch.setattr(cli, "port_in_use", lambda host, port: True)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: started.append(kwargs))
    monkeypatch.setenv("LOCALDEPLOY_HOME", str(__import__("tempfile").mkdtemp()))

    assert cli.main(["--port", "8000", "--no-browser"]) == 2
    assert started == []


def test_port_in_use_sees_a_listening_socket():
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        busy_port = listener.getsockname()[1]
        assert cli.port_in_use("127.0.0.1", busy_port) is True
    assert cli.port_in_use("127.0.0.1", busy_port) is False
