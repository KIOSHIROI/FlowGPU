"""The lexical stage on its own.

There is exactly one grammar. This lexer is the parser's grammar with the
parser left out (`parser=None`), so the keyword and dtype sets cannot drift
from the ones the parser uses -- the duplication that used to live in a
second grammar file plus two Python dictionaries is gone.

Lark's LALR parser lexes *contextually* by default: it tells a keyword from
an identifier by consulting the parser state. A standalone lexer has no such
context, so grammar.lark gives DTYPE an explicit priority (DTYPE.2) to make
both agree.
"""

from pathlib import Path

from lark import Lark

GRAMMAR_FILE = Path(__file__).resolve().parents[1] / "grammar.lark"

lexer = Lark.open(
    str(GRAMMAR_FILE),
    parser=None,
    lexer="basic",
)


def tokenize(source: str):
    """Yield the tokens of `source`, with comments and whitespace removed."""
    yield from lexer.lex(source)
