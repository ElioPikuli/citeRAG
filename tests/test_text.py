from citerag.text import split_sentences, stem, tokenize


def test_splits_camel_and_snake_case_and_keeps_whole_identifier():
    assert tokenize("calcLaborDays") == ["calclaborday", "calc", "labor", "day"]
    assert tokenize("include_waste") == ["include_waste", "include", "waste"]
    assert "m3perworkerday" in tokenize("m3PerWorkerDay")


def test_drops_stopwords_and_single_letters_but_keeps_digits():
    assert tokenize("What is the price of a 5 m slab") == ["price", "5", "slab"]
    assert "the" in tokenize("the price", keep_stopwords=True)


def test_plural_folding_is_light():
    assert stem("defaults") == "default"
    assert stem("queries") == "query"
    assert stem("matches") == "match"
    assert stem("class") == "class"  # -ss untouched
    assert stem("status") == "status"  # -us untouched
    assert stem("bus") == "bus"  # too short


def test_unicode_words_tokenize():
    assert tokenize("מחיר הבטון למ״ק") == ["מחיר", "הבטון", "למק"]
    assert tokenize("מע\"מ") == ["מעמ"]


def test_hebrew_sentence_split():
    assert split_sentences("משפט ראשון. משפט שני.") == ["משפט ראשון.", "משפט שני."]


def test_sentence_split():
    assert split_sentences("One thing. Two things!\n\nThird") == ["One thing.", "Two things!", "Third"]
