import pytest
from fastapi.testclient import TestClient

from ragkit.api.auth import limiter
from ragkit.api.main import Components, create_app


@pytest.fixture
def client(settings, store, service, indexed):
    limiter.reset()
    app = create_app(Components(settings, store, service, indexed))
    return TestClient(app, raise_server_exceptions=False)


EMP = {"X-API-Key": "emp-key"}
ADMIN = {"X-API-Key": "admin-key"}


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["ready"] is True


def test_auth_required(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 401
    assert (
        client.post("/query", json={"question": "hi"}, headers={"X-API-Key": "nope"}).status_code
        == 401
    )


def test_query_returns_role_and_citations(client):
    r = client.post(
        "/query", json={"question": "How many PTO days per calendar year?"}, headers=EMP
    )
    body = r.json()
    assert r.status_code == 200 and body["role"] == "employee"
    assert body["citations"] and "debug" not in body
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_security_headers(client):
    r = client.get("/")
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Frame-Options"] == "DENY"


def test_ingest_requires_admin(client):
    assert client.post("/ingest/sync", headers=EMP).status_code == 403
    r = client.post("/ingest/sync?dry_run=true", headers=ADMIN)
    assert r.status_code == 200 and r.json()["plan"]["unchanged"] == 21


def test_upload_validation(client):
    def up(name, data):
        return client.post("/ingest/upload", headers=ADMIN, files={"file": (name, data)})

    assert up("../../etc/passwd.md", b"x").status_code in (200, 400)  # basename only
    assert up(".hidden.md", b"x").status_code == 400
    assert up("evil.pdf", b"MZ not a pdf").status_code == 415
    assert up("tool.exe", b"MZ").status_code == 415
    ok = up("faq-extra.md", b"# Extra FAQ\nThe cafeteria opens at 8am.")
    assert ok.status_code == 200 and ok.json()["plan"]["new"] == 1


def test_rate_limit(client, settings):
    settings.rate_limit_per_minute = 3
    codes = [client.get("/ingest/status", headers=ADMIN).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
