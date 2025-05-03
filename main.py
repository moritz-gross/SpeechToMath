#!/usr/bin/env python3
"""
voice_calc.py  –  Transcribe spoken math (EN/DE) with Whisper
                  and convert it to a SymPy expression.
"""
import re
import whisper
from lark import Lark, Transformer, v_args, exceptions
import sympy as sp


# ----------------------------------------------------------------------
# 1)  Speech‑to‑text
# ----------------------------------------------------------------------
def transcribe(audio_path: str, model_size: str = "base") -> str:
    """
    Run Whisper locally and return lower‑cased transcription.
    Adjust 'model_size' to tiny|base|small|medium|large  as you need.
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language=None)   # auto‑detect EN / DE
    return result["text"].strip().lower()


# ----------------------------------------------------------------------
# 2)  NL‑to‑Math grammar (English + German)
#     – minimal but extendable
# ----------------------------------------------------------------------
GRAMMAR = r"""
    ?start: expr

    // ----- operator precedence (bottom = highest)  -----
    ?expr: expr PLUS  term   -> add
         | expr MINUS term   -> sub
         | term

    ?term: term TIMES  power -> mul
         | term DIV   power  -> div
         | power

    ?power: atom POW power   -> pow
          | atom

    // ----- atoms -----
    ?atom: FUNC atom         -> func
         | SQRT atom         -> sqrt
         | NUMBER            -> number
         | VAR               -> var
         | LPAR expr RPAR    -> parens

    // ----- tokens -----
    PLUS:  "plus" | "add" | "und"  // German "und" (optional)
    MINUS: "minus" | "subtract" | "abziehen"
    TIMES: "times" | "multiplied by" | "mal"
    DIV:   "divided by" | "over" | "durch" | "geteilt durch"
    POW:   ("to the power of" | "hoch")
    SQRT:  "square root of"  | "wurzel aus"
    FUNC:  /(sin|sine|cos|cosine|tan|tangent|arcsin|arccos|arctan|log|ln|exp|abs|sinus|cosinus)/
    LPAR:  "open parenthesis" | "open bracket" | "klammer auf"
    RPAR:  "close parenthesis" | "close bracket" | "klammer zu"
    %import common.SIGNED_NUMBER   -> NUMBER
    VAR:   /[a-z]\w*/i

    %ignore /\s+/
"""

PARSER = Lark(GRAMMAR, parser="lalr", propagate_positions=False)


# ----------------------------------------------------------------------
# 3)  Tree → SymPy
# ----------------------------------------------------------------------
@v_args(inline=True)
class ToSymPy(Transformer):
    number = lambda self, tok: sp.Number(tok)
    var    = lambda self, tok: sp.symbols(tok)
    add    = lambda self, a, b: a + b
    sub    = lambda self, a, b: a - b
    mul    = lambda self, a, b: a * b
    div    = lambda self, a, b: a / b
    pow    = lambda self, a, b: a ** b
    sqrt   = lambda self, x: sp.sqrt(x)
    parens = lambda self, x: x

    def func(self, name_tok, arg):
        name = str(name_tok)
        mapping = {
            # English → SymPy
            "sin": sp.sin, "sine": sp.sin,
            "cos": sp.cos, "cosine": sp.cos,
            "tan": sp.tan, "tangent": sp.tan,
            "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
            "log": sp.log,  # natural log unless base given
            "ln": sp.log,
            "exp": sp.exp,
            "abs": sp.Abs,
            # German synonyms
            "sinus": sp.sin,
            "cosinus": sp.cos,
        }
        return mapping[name](arg)


def nl_to_sympy(text: str):
    """
    Parse a natural‑language math string to a SymPy expression object.
    Raises ValueError if parsing fails.
    """
    try:
        tree = PARSER.parse(text)
        return ToSymPy().transform(tree)
    except exceptions.LarkError as e:
        raise ValueError(f"Could not parse: {text!r}") from e


# ----------------------------------------------------------------------
# 4)  Example CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="Voice calculator demo")
    ap.add_argument("audio", help="Path to an audio file (wav/mp3/ogg…)")
    ap.add_argument("-m", "--model", default="base", help="Whisper model size")
    args = ap.parse_args()

    # 1) STT ------------------------------------------------------------
    utterance = transcribe(args.audio, args.model)
    print(f"[Transcription] {utterance}")

    # 2) Natural‑language  → SymPy -------------------------------------
    try:
        expr = nl_to_sympy(utterance)
    except ValueError as err:
        print("Sorry – couldn't understand that:", err, file=sys.stderr)
        sys.exit(1)

    # 3) Evaluate & speak‑back -----------------------------------------
    #    For demo we just print; integrate a TTS engine for speech output.
    print(f"[Parsed]       {expr}")
    try:
        numeric = expr.evalf()
        print(f"[Result]       {numeric}")
    except (TypeError, sp.SympifyError):
        print("[Note] Symbolic expression only – no numeric value")