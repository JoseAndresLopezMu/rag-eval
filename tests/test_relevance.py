import pytest

from rag_eval.models import Context, Evidence
from rag_eval.relevance import covers, normalize, quote_coverage

# Excerpts from the real tender document used by chatbot-rag
# (pliego_tecnico_madrimasd_S_2026_003.pdf, page 7).
SOURCE = "pliego_tecnico_madrimasd_S_2026_003.pdf"
QUOTE = "Servicio de recuperación de datos"

CHUNK_WITH_QUOTE = (
    "Fundación y mejorar el trabajo en grupo sobre documentos compartidos al seguir "
    "accesible el sistema de bloqueos que ofrece CIFS. • Servicio de recuperación de "
    "datos. • Soporte in situ a usuarios."
)
# The cover page shares most of the quote's words, but not the phrase itself.
# A bag-of-words rule scores it at 80 %.
COVER_PAGE_CHUNK = (
    "PLIEGO DE PRESCRIPCIONES TÉCNICAS QUE HAN DE REGIR EL CONCURSO POR EL "
    "PROCEDIMIENTO ABIERTO SIMPLIFICADO CON PLURALIDAD DE CRITERIOS PARA LA "
    "CONTRATACIÓN DE LOS SERVICIOS DE MANTENIMIENTO, SOPORTE Y RECUPERACIÓN DE "
    "SISTEMAS Y DATOS"
)

LONG_QUOTE = (
    "será preciso, por parte del adjudicatario, implementar las siguientes tareas: "
    "realización de visitas presenciales de soporte correctivo y preventivo"
)
# Same passage split by a chunk boundary: only the first 12 of 19 tokens fit.
CHUNK_CUT_MID_QUOTE = (
    "4.5 SERVICIOS DE MANTENIMIENTO COMUNES Para todos los servicios de mantenimiento "
    "que sean contratados será preciso, por parte del adjudicatario, implementar las "
    "siguientes tareas: realización de"
)


def context(text: str, source: str = SOURCE) -> Context:
    return Context(rank=1, source=source, text=text)


class TestNormalize:
    def test_lowercases_and_strips_accents_and_punctuation(self) -> None:
        assert normalize("Recuperación, DATOS; Madri+d!") == ["recuperacion", "datos", "madri", "d"]

    def test_empty_text(self) -> None:
        assert normalize(" .. ") == []


class TestQuoteCoverage:
    def test_full_quote_present(self) -> None:
        assert quote_coverage(QUOTE, CHUNK_WITH_QUOTE) == 1.0

    def test_shared_words_without_the_phrase_score_low(self) -> None:
        assert quote_coverage(QUOTE, COVER_PAGE_CHUNK) < 0.6

    def test_quote_cut_by_chunk_boundary(self) -> None:
        assert quote_coverage(LONG_QUOTE, CHUNK_CUT_MID_QUOTE) == pytest.approx(12 / 19)

    def test_quote_without_words_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no words"):
            quote_coverage("...", CHUNK_WITH_QUOTE)


class TestCovers:
    def test_chunk_with_quote(self) -> None:
        assert covers(context(CHUNK_WITH_QUOTE), Evidence(source=SOURCE, quote=QUOTE))

    def test_cover_page_is_not_relevant(self) -> None:
        assert not covers(context(COVER_PAGE_CHUNK), Evidence(source=SOURCE, quote=QUOTE))

    def test_cut_quote_counts_at_default_threshold(self) -> None:
        assert covers(context(CHUNK_CUT_MID_QUOTE), Evidence(source=SOURCE, quote=LONG_QUOTE))

    def test_cut_quote_fails_with_stricter_threshold(self) -> None:
        evidence = Evidence(source=SOURCE, quote=LONG_QUOTE)
        assert not covers(context(CHUNK_CUT_MID_QUOTE), evidence, threshold=0.7)

    def test_threshold_is_inclusive(self) -> None:
        evidence = Evidence(source=SOURCE, quote="uno dos tres cuatro cinco")
        assert covers(context("uno dos tres"), evidence, threshold=0.6)

    def test_different_source_never_covers(self) -> None:
        evidence = Evidence(source="oferta_nubesoft_iberica.docx", quote=QUOTE)
        assert not covers(context(CHUNK_WITH_QUOTE), evidence)
