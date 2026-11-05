from citerag.loaders import html_to_text_preserving_lines, load_documents

HTML = """<html>
<head><style>
p { color: red; }
</style></head>
<body>
<h2 class="x">Title &amp; more</h2>
<p
  class="multi">Hello <b>world</b></p>
<script>
var x = 1;
</script>
<!-- a
comment -->
</body></html>"""


def test_html_keeps_line_count_and_marks_headings():
    out = html_to_text_preserving_lines(HTML)
    lines = out.split("\n")
    assert len(lines) == HTML.count("\n") + 1
    assert lines[5] == "## Title & more"  # same line number as in the source
    assert "Hello world" in out
    assert "color" not in out and "var x" not in out and "comment" not in out


def test_load_documents_skips_excluded_dirs_and_binaries(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.md").write_text("# hi\ntext")
    (tmp_path / "a" / "node_modules").mkdir()
    (tmp_path / "a" / "node_modules" / "y.js").write_text("x")
    (tmp_path / "a" / "bin.js").write_bytes(b"\x00\x01binary")
    (tmp_path / "a" / "skip.png").write_bytes(b"png")
    docs = load_documents([tmp_path / "a"], (".md", ".js"), ("node_modules",))
    assert [d.source for d in docs] == ["a/x.md"]
    assert len(docs[0].sha256) == 64


def test_missing_source_is_skipped(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.md").write_text("# hi\ntext")
    docs = load_documents([tmp_path / "missing", tmp_path / "a"], (".md",), ())
    assert [d.source for d in docs] == ["a/x.md"]
    assert "not found, skipping" in capsys.readouterr().err
