"""Document cleaning helpers used before NER inference.

This is the preprocessing step the project applies to every clinical
note before running the entity recogniser. It lives here so both
``ner/extract_entities.py`` (which then runs NER and assertion
classification) and ``ner/clean_documents.py`` (which writes the
cleaned text on its own) produce the same manuscript-faithful input.

The cleaning pipeline removes mid-sentence hard line wraps, normalises
whitespace, runs PyRuSH sentence splitting with the medspaCy section
detector, treats each section title as a sentence boundary (even when
PyRuSH placed it inside a sentence), and merges sentences that PyRuSH
split after a stop word likely to continue onto the next line.

Two entry points are exposed: ``clean_document_text`` returns just the
joined cleaned note, and ``clean_document_sentences`` returns the same
text plus the per-sentence list and character offsets used downstream
to map NER spans back into the cleaned document. Both accept a
pre-built spaCy pipeline (from ``build_cleaning_nlp``) so callers
processing many documents can build the pipeline once per worker
instead of once per call.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import spacy
from spacy.language import Language
from medspacy.section_detection import Sectionizer
from medspacy.sentence_splitting import PyRuSHSentencizer


SCRIPT_DIR = Path(__file__).resolve().parent
RUSH_RULES_PATH = SCRIPT_DIR / "rush_rules.tsv"
SECTION_PATTERNS_PATH = SCRIPT_DIR / "section_patterns.json"

# A newline preceded by any character and followed by a non-whitespace
# character; i.e., a hard line wrap that interrupts a sentence rather
# than a paragraph break.
PATTERN_MIDLINE_NEWLINES = re.compile(r"(?<=\S|\s)\n(?=\S)")

# Function words that frequently appear at the end of a PyRuSH-split
# line when the sentence actually continues onto the next line. Kept
# module-level so the set is built once rather than per call.
_DEFAULT_STOP_WORDS: frozenset[str] = frozenset({
    # Articles and determiners
    "a", "an", "the", "my", "your", "his", "her", "its", "their", "our", "one's",
    "this", "that", "these", "those", "some", "any", "enough", "many", "much",
    "few", "several", "all", "both", "each", "every", "either", "neither", "none",
    "plenty", "most",
    # Prepositions
    "in", "on", "at", "by", "to", "of", "for", "with", "from", "about", "like",
    "through", "over", "under", "between", "before", "after", "around", "near",
    "inside", "outside", "within", "without", "during", "against",
    # Conjunctions
    "and", "or", "but", "nor", "yet", "so",
    # Subordinating conjunctions
    "if", "since", "because", "although", "though", "when", "while", "until",
    "where", "unless", "as", "than", "once", "whether",
    # Modal/Auxiliary verbs
    "can", "could", "would", "should", "must", "may", "might",
    "is", "are", "was", "were", "has", "have", "had", "do", "does", "did",
    "will", "shall", "been",
    # Pronouns
    "he", "she", "it", "we", "you", "they", "one", "who", "whom",
    "whose", "what", "which", "someone", "something", "anyone",
    "anything", "everyone", "everything",
    # Adverbs
    "here", "there", "then", "so", "also", "too", "only", "still",
    "just", "even", "yet", "well",
    # Common contractions
    "it's", "he's", "she's", "they're", "we're", "you're", "I've",
    "we've", "they've", "he'd", "she'd", "they'd",
})


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space."""
    return " ".join(text.split())


def reconnect_stop_word_splits(
    sentences: list[str],
    stop_words: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Merge sentences that PyRuSH split after a stop word.

    PyRuSH occasionally ends a sentence on a function word (article,
    preposition, conjunction, auxiliary, etc.) when the next line begins
    with a capital letter. When that happens, treat the two lines as one
    sentence rather than two, since the stop word almost certainly leads
    into the next clause.
    """
    if stop_words is None:
        stop_words = _DEFAULT_STOP_WORDS

    merged_sentences: list[str] = []
    i = 0
    while i < len(sentences):
        current_sent = sentences[i].strip()
        if i < len(sentences) - 1:
            next_sent = sentences[i + 1].strip()
            last_word = current_sent.split()[-1].lower() if current_sent else ""
            first_char = next_sent[0] if next_sent else ""
            if last_word in stop_words and first_char.isupper():
                merged_sentences.append(current_sent + " " + next_sent)
                i += 2
                continue
        merged_sentences.append(current_sent)
        i += 1
    return merged_sentences


def build_cleaning_nlp() -> Language:
    """Construct a blank-en spaCy pipeline with PyRuSH and a section detector.

    The PyRuSH and Sectionizer factories are registered under
    UUID-suffixed names so several pipelines can be built in the same
    Python process (for example, one pipeline per worker plus one in the
    parent process for ad hoc cleaning) without spaCy raising a
    duplicate-factory error.
    """
    nlp = spacy.blank("en")

    splitter_name = f"custom_splitter_{uuid.uuid4().hex[:8]}"
    sectionizer_name = f"custom_sectionizer_{uuid.uuid4().hex[:8]}"

    @Language.factory(splitter_name)
    def _create_custom_splitter(nlp, name):
        return PyRuSHSentencizer(nlp=nlp, rules_path=str(RUSH_RULES_PATH))

    nlp.add_pipe(splitter_name)

    @Language.factory(sectionizer_name)
    def _create_custom_sectionizer(nlp, name):
        return Sectionizer(nlp, rules=str(SECTION_PATTERNS_PATH))

    nlp.add_pipe(sectionizer_name, last=True)
    return nlp


def clean_document_sentences(
    text: str,
    nlp: Language | None = None,
) -> tuple[str, list[str], list[int]]:
    """Run the full cleaning pipeline and return per-sentence offsets.

    Returns ``(new_doc, cleaned_sentences, sentence_starts)``:

    * ``new_doc`` is the cleaned sentences joined with newlines.
    * ``cleaned_sentences`` is the list of cleaned sentence strings.
    * ``sentence_starts`` is each sentence's character offset within
      ``new_doc`` so callers can map per-sentence NER spans back into
      cleaned-document coordinates.

    A section title that PyRuSH placed inside a sentence is treated as a
    sentence boundary, since the title and the body underneath it are
    rarely a single semantic unit. When ``nlp`` is omitted, a fresh
    pipeline is built per call; pass one in for batch processing.
    """
    if nlp is None:
        nlp = build_cleaning_nlp()

    text_proc = text.strip()
    doc_sentence_splitted = nlp(text_proc)
    sections = doc_sentence_splitted._.sections

    raw_sentences: list[str] = []
    for sent in doc_sentence_splitted.sents:
        sent_start = sent.start
        sent_end = sent.end
        splits = [sent_start]
        for section in sections:
            title_start, _ = section.title_span
            if sent_start < title_start < sent_end:
                splits.append(title_start)
        splits.append(sent_end)
        splits = sorted(set(splits))
        for i in range(len(splits) - 1):
            span = doc_sentence_splitted[splits[i]:splits[i + 1]]
            target = PATTERN_MIDLINE_NEWLINES.sub(" ", span.text).strip()
            target = normalize_whitespace(target)
            if target:
                raw_sentences.append(target)

    fixed_sentences = reconnect_stop_word_splits(raw_sentences)

    cleaned_sentences: list[str] = []
    sentence_starts: list[int] = []
    current_pos = 0
    new_doc = ""
    for sent in fixed_sentences:
        cleaned_sentences.append(sent)
        sentence_starts.append(current_pos)
        current_pos += len(sent) + 1  # +1 for the joining newline
        new_doc += ("\n" if new_doc else "") + sent
    return new_doc, cleaned_sentences, sentence_starts


def clean_document_text(text: str, nlp: Language | None = None) -> str:
    """Return only the joined cleaned text from ``clean_document_sentences``."""
    new_doc, _, _ = clean_document_sentences(text, nlp=nlp)
    return new_doc
