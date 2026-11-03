from ragkit.ingestion.chunker import chunk_document, count_tokens, split_sections
from ragkit.ingestion.normalize import normalize_text


def test_normalize_strips_invisible_and_tag_characters():
    smuggled = "".join(chr(0xE0000 + ord(c)) for c in "evil")
    raw = "Ig​nore‮ this\x07 " + smuggled + "\r\nnext\n\n\n\nline"
    n = normalize_text(raw)
    assert n.text == "Ignore this\nnext\n\nline"
    assert n.hidden_chars_removed == 2 + len("evil")


def test_normalize_nfkc_folds_lookalikes():
    assert normalize_text("ｉｇｎｏｒｅ").text == "ignore"  # full-width letters


def test_split_sections_tracks_heading_path():
    text = "# Title\nintro\n## A\na text\n### A1\na1 text\n## B\nb text"
    secs = split_sections(text)
    assert secs == [
        (["Title"], "intro"),
        (["Title", "A"], "a text"),
        (["Title", "A", "A1"], "a1 text"),
        (["Title", "B"], "b text"),
    ]


def test_chunks_have_breadcrumb_header_and_respect_budget():
    para = "Sentence about the leave policy and its many rules. " * 30
    text = "# Leave\n## Rules\n" + "\n\n".join([para] * 4)
    chunks = chunk_document(text, "Leave", max_tokens=200, overlap=30)
    assert len(chunks) > 3
    for c in chunks:
        assert c.text.startswith("Leave > Rules\n\n")
        assert count_tokens(c.text) <= 200 + 5
        assert c.section == "Rules"  # duplicated H1 title is dropped from the path


def test_chunk_overlap_carries_context():
    paras = [f"Paragraph {i} " + "word " * 20 for i in range(12)]
    chunks = chunk_document("\n\n".join(paras), "Doc", max_tokens=120, overlap=40)
    assert len(chunks) >= 2
    first_last_para = chunks[0].body.split("\n\n")[-1]
    assert first_last_para in chunks[1].body


def test_empty_document_gives_no_chunks():
    assert chunk_document("   \n\n", "Empty", 200, 20) == []
