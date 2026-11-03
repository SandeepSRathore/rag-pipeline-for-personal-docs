from pathlib import Path

import pytest

from ragkit.embeddings.providers import FakeEmbedder
from ragkit.ingestion.manifest import Manifest
from ragkit.ingestion.pipeline import Ingestor
from ragkit.store.qdrant_store import PipelineVersionMismatchError
from tests.conftest import FakeSparse


def _points_for(store, path: str) -> list[dict]:
    pts, _ = store.client.scroll(store.alias, limit=1000, with_payload=True)
    return [p.payload for p in pts if p.payload["path"] == path]


def test_first_ingest_indexes_everything(indexed: Ingestor, store):
    entries = Manifest(indexed.settings.db_path).all_entries()
    assert len(entries) == 21
    assert all(e.status == "ok" for e in entries)
    assert store.count() == sum(len(e.chunk_ids) for e in entries)


def test_second_run_is_a_noop(indexed: Ingestor, corpus: Path):
    rep = indexed.ingest(corpus, source="corpus")
    assert rep.plan == {"new": 0, "changed": 0, "unchanged": 21, "deleted": 0}
    assert rep.embedded_via_api == 0 and rep.chunks_written == 0


def test_edit_reembeds_only_changed_chunks(indexed: Ingestor, corpus: Path, store):
    p = corpus / "hr" / "pto-policy-2025.md"
    p.write_text(p.read_text().replace("Up to 10 unused", "Up to 12 unused"))
    dry = indexed.ingest(corpus, source="corpus", dry_run=True)
    assert dry.plan["changed"] == 1 and dry.planned["changed"] == ["hr/pto-policy-2025.md"]

    rep = indexed.ingest(corpus, source="corpus")
    assert rep.embedded_via_api == 1  # only the "Carryover" chunk changed
    assert rep.embedding_cache_hits >= 5
    texts = [pl["text"] for pl in _points_for(store, "hr/pto-policy-2025.md")]
    assert any("Up to 12 unused" in t for t in texts)
    assert not any("Up to 10 unused" in t for t in texts)  # stale chunk deleted


def test_new_and_deleted_files(indexed: Ingestor, corpus: Path, store):
    (corpus / "hr" / "travel-policy.md").unlink()
    (corpus / "hr" / "new-doc.md").write_text("---\ntitle: New Doc\n---\n# New Doc\nfresh text")
    rep = indexed.ingest(corpus, source="corpus")
    assert rep.plan["new"] == 1 and rep.plan["deleted"] == 1
    assert _points_for(store, "hr/travel-policy.md") == []
    assert len(_points_for(store, "hr/new-doc.md")) == 1


def test_metadata_only_change_updates_payload(indexed: Ingestor, corpus: Path, store):
    meta = corpus / "product" / "pricing-tiers.html.meta.yaml"
    meta.write_text(meta.read_text().replace("acl: public", "acl: hr"))
    rep = indexed.ingest(corpus, source="corpus")
    assert rep.plan["changed"] == 1 and rep.embedded_via_api == 0
    assert {pl["acl"] for pl in _points_for(store, "product/pricing-tiers.html")} == {"hr"}


def test_broken_update_keeps_last_good_version(indexed: Ingestor, corpus: Path, store):
    before = _points_for(store, "hr/expense-policy.pdf")
    (corpus / "hr" / "expense-policy.pdf").write_bytes(b"not a pdf anymore")
    rep = indexed.ingest(corpus, source="corpus")
    assert rep.failed and rep.failed[0]["path"] == "hr/expense-policy.pdf"
    assert _points_for(store, "hr/expense-policy.pdf") == before
    # failed docs are retried on the next run
    assert indexed.ingest(corpus, source="corpus", dry_run=True).plan["changed"] == 1


def test_injected_chunk_is_quarantined(indexed: Ingestor, store):
    flagged = store.flagged_chunks()
    assert len(flagged) == 1
    assert flagged[0]["path"] == "vendors/vendor-notes.md"


def test_pipeline_change_requires_reindex(indexed: Ingestor, corpus: Path, settings, store):
    settings.chunk_tokens = 300  # changes pipeline_version
    ing2 = Ingestor(settings, store=store, dense=FakeEmbedder(), sparse=FakeSparse())
    with pytest.raises(PipelineVersionMismatchError):
        ing2.ingest(corpus, source="corpus")
    old = store.alias_target()
    res = ing2.reindex()
    assert store.alias_target() == res["new_collection"] != old
    assert not store.client.collection_exists(old)
    assert ing2.ingest(corpus, source="corpus").plan["unchanged"] == 21
