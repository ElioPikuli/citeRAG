import pytest

from citerag.chunking import chunk_document
from citerag.loaders import Document


def doc(text: str, source: str = "p/readme.md") -> Document:
    return Document(path=__import__("pathlib").Path(source), source=source, text=text, sha256="x")


def test_markdown_sections_get_heading_paths_and_true_line_numbers():
    text = "# Top\nintro\n\n## Labor\nlabor text\n\n### Detail\ndetail text\n## Other\nother"
    chunks = chunk_document(doc(text))
    titles = [c.title for c in chunks]
    assert titles == ["Top", "Top > Labor", "Top > Labor > Detail", "Top > Other"]
    lines = text.split("\n")
    for c in chunks:
        # every non-blank line of the chunk is the source line at that position
        assert lines[c.start_line - 1] in c.text and lines[c.end_line - 1] in c.text


def test_heading_inside_code_fence_is_not_a_section():
    text = "# Doc\n```bash\n# not a heading\necho hi\n```\nafter"
    chunks = chunk_document(doc(text))
    assert len(chunks) == 1 and "# not a heading" in chunks[0].text


def test_heading_only_sections_are_dropped():
    chunks = chunk_document(doc("# A\n## B\nbody"))
    assert [c.title for c in chunks] == ["A > B"]


def test_size_limit_and_overlap():
    # paragraphs of short lines, so whole trailing lines fit in the overlap budget
    para = "\n\n".join("\n".join(f"p{i} line {j} " + "word " * 5 for j in range(4)) for i in range(12))
    chunks = chunk_document(doc("# T\n" + para), max_chars=400, overlap_chars=120)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 for c in chunks)
    # consecutive chunks share at least one line (overlap) and move forward
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_line <= a.end_line < b.end_line


def test_long_single_line_is_hard_split():
    chunks = chunk_document(doc("x" * 1000), max_chars=300, overlap_chars=0)
    assert all(len(c.text) <= 300 for c in chunks)
    assert "".join(c.text for c in chunks) == "x" * 1000


def test_code_is_split_per_function_with_leading_comment():
    code = (
        "(function () {\n"
        "  // Rounds up to whole days.\n"
        "  function calcLaborDays(v) {\n    return Math.ceil(v);\n  }\n\n"
        "  function calcVat(a) {\n    return a * 0.18;\n  }\n"
        "})();\n"
    )
    chunks = chunk_document(doc(code, "p/calc.js"))
    by_title = {c.title: c for c in chunks}
    assert "calc.js > calcLaborDays" in by_title and "calc.js > calcVat" in by_title
    assert "Rounds up to whole days" in by_title["calc.js > calcLaborDays"].text
    assert by_title["calc.js > calcLaborDays"].text.startswith("  //")  # indentation kept


def test_chunk_ids_are_stable():
    a = chunk_document(doc("# T\nsame text"))
    b = chunk_document(doc("# T\nsame text"))
    assert [c.id for c in a] == [c.id for c in b]


def test_overlap_must_be_smaller_than_max():
    with pytest.raises(ValueError):
        chunk_document(doc("x"), max_chars=100, overlap_chars=100)


def test_csharp_is_split_per_class_and_method():
    code = (
        "using UnityEngine;\n\n"
        "public class Pathfinder : MonoBehaviour\n{\n"
        "    public float epsilon = 2.5f;\n\n"
        "    /// <summary>Weighted A*.</summary>\n"
        "    public List<Node> FindPath(Vector3 start, Vector3 target)\n    {\n"
        "        if (start == target)\n            return Retrace(start);\n"
        "        foreach (var n in open)\n        {\n        }\n        return null;\n    }\n\n"
        "    [ContextMenu(\"Reset\")]\n"
        "    private IEnumerator ResetNodes()\n    {\n        yield return null;\n    }\n}\n"
    )
    chunks = chunk_document(doc(code, "game/Pathfinder.cs"))
    titles = [c.title for c in chunks]
    assert titles == ["Pathfinder.cs", "Pathfinder.cs > Pathfinder",
                      "Pathfinder.cs > Pathfinder.FindPath", "Pathfinder.cs > Pathfinder.ResetNodes"]
    by_title = {c.title: c for c in chunks}
    assert "Weighted A*" in by_title["Pathfinder.cs > Pathfinder.FindPath"].text
    assert "ContextMenu" in by_title["Pathfinder.cs > Pathfinder.ResetNodes"].text
