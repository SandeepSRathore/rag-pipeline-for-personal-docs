import pytest
from qdrant_client import models

from lcrag.indexing import LCIndexer
from lcrag.vectorstore import PipelineVersionMismatchError


def _payloads(ix: LCIndexer, source: str) -> list[dict]:
    pts, _ = ix.client.scroll(
        ix.collections.alias,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))
            ]
        ),
        limit=1000,
        with_payload=True,
    )
    return [p.payload for p in pts]


def test_first_ingest_adds_everything(indexer, settings):
    rep = indexer.ingest(settings.corpus_dir, source="corpus")
    assert rep.plan["new"] == 21
    assert rep.indexing_result["num_added"] > 80
    assert rep.flagged_chunks == 1


def test_second_run_skips_everything(indexed, settings):
    rep = indexed.ingest(settings.corpus_dir, source="corpus")
    assert rep.plan == {"new": 0, "changed": 0, "unchanged": 21, "deleted": 0}
    assert rep.indexing_result["num_added"] == 0 and rep.indexing_result["num_deleted"] == 0
    assert rep.embedded_via_api == 0


def test_edit_reembeds_only_changed_chunk(indexed, settings):
    p = settings.corpus_dir / "hr" / "pto-policy-2025.md"
    p.write_text(p.read_text().replace("Up to 10 unused", "Up to 12 unused"))
    dry = indexed.ingest(settings.corpus_dir, source="corpus", dry_run=True)
    assert dry.planned["changed"] == ["hr/pto-policy-2025.md"]
    rep = indexed.ingest(settings.corpus_dir, source="corpus")
    assert rep.indexing_result["num_added"] == 1 and rep.indexing_result["num_deleted"] == 1
    assert rep.embedded_via_api == 1
    texts = [pl["page_content"] for pl in _payloads(indexed, "hr/pto-policy-2025.md")]
    assert any("Up to 12" in t for t in texts) and not any("Up to 10" in t for t in texts)


def test_deleted_and_new_files(indexed, settings):
    (settings.corpus_dir / "hr" / "travel-policy.md").unlink()
    (settings.corpus_dir / "hr" / "new.md").write_text("---\ntitle: New\n---\n# New\nhello")
    rep = indexed.ingest(settings.corpus_dir, source="corpus")
    assert rep.plan["new"] == 1 and rep.plan["deleted"] == 1
    assert _payloads(indexed, "hr/travel-policy.md") == []
    assert len(_payloads(indexed, "hr/new.md")) == 1


def test_metadata_change_rewrites_payload_without_reembedding(indexed, settings):
    meta = settings.corpus_dir / "product" / "pricing-tiers.html.meta.yaml"
    meta.write_text(meta.read_text().replace("acl: public", "acl: hr"))
    rep = indexed.ingest(settings.corpus_dir, source="corpus")
    assert rep.indexing_result["num_added"] > 0
    assert rep.embedded_via_api == 0  # CacheBackedEmbeddings: same text, cached vector
    assert {pl["metadata"]["acl"] for pl in _payloads(indexed, "product/pricing-tiers.html")} == {
        "hr"
    }


def test_broken_file_keeps_last_good_version(indexed, settings):
    before = _payloads(indexed, "hr/expense-policy.pdf")
    (settings.corpus_dir / "hr" / "expense-policy.pdf").write_bytes(b"not a pdf")
    rep = indexed.ingest(settings.corpus_dir, source="corpus")
    assert rep.failed[0]["path"] == "hr/expense-policy.pdf"
    assert rep.plan["deleted"] == 0
    assert _payloads(indexed, "hr/expense-policy.pdf") == before


def test_settings_change_requires_reindex(indexed, settings):
    settings.chunk_tokens = 300
    ix2 = LCIndexer(settings, client=indexed.client)
    with pytest.raises(PipelineVersionMismatchError):
        ix2.ingest(settings.corpus_dir, source="corpus")
    old = ix2.collections.target()
    res = ix2.reindex()
    assert ix2.collections.target() == res["new_collection"] != old
    assert not indexed.client.collection_exists(old)
    assert ix2.ingest(settings.corpus_dir, source="corpus").plan["unchanged"] == 21
