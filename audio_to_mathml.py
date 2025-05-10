from sympy.parsing.sympy_parser import parse_expr
import openai
import os
import argparse
from typing import List, Dict
import sympy as sp

openai.api_key = os.environ.get("OPENAI_API_KEY")

class InvalidSympyException(Exception):
    pass


def transcribe(audio_path: str) -> List[Dict]:
    with open(audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    return [
        {
            "word": w.word,
            "start": round(w.start, 2),
            "end": round(w.end, 2)
        }
        for w in transcript.words]


def text_to_sympy(txt: str):
    examples = [
        {"text": "one plus two", "expression": "1 + 2"},
        {"text": "the square root of ten minus five", "expression": "sqrt(10) - 5"}
    ]
    examples_text = "\n".join([f"Text: {ex['text']}\nExpression: {ex['expression']}" for ex in examples])

    prompt = f"""Convert the following natural‑language description of a mathematical expression
    into a valid SymPy expression **without using any module prefix** (e.g. use `cos(x)` not `sp.cos(x)`).

    Here are some examples:

    {examples_text}

    Text: {txt}
    Expression:"""

    response = openai.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system",
            "content": "You are a helpful assistant that converts natural language math descriptions into sympy expressions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=100
    )

    expression_text = response.choices[0].message.content.strip()
    try:
        return parse_expr(expression_text, evaluate=False)
    except:
        raise InvalidSympyException("input could not be parsed as a sympy expression")

def audio_to_mathml(path: str):
    transcription = transcribe(path)
    plain_text = " ".join([w["word"] for w in transcription])
    print(f"[raw transcription] {plain_text}")

    sympy_expr = text_to_sympy(plain_text)
    print(f"[sympy expression] {sympy_expr}")

    return sp.printing.mathml(sympy_expr) # use printer='presentation' if you want presentation mathml



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to an audio file (wav/mp3/ogg…)")
    args = ap.parse_args()

    result = audio_to_mathml(path=args.audio)
    print(result)