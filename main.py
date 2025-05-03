from sympy.parsing.sympy_parser import parse_expr
import openai
import os
import argparse
from typing import List, Dict

openai.api_key = os.environ.get("OPENAI_API_KEY")


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
    Use the word level timestamps to infer which parts to group together in the expression.

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
    return parse_expr(expression_text, evaluate=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to an audio file (wav/mp3/ogg…)")
    args = ap.parse_args()

    described_text: str = transcribe(args.audio)
    print(f"[raw transcription] {' '.join([x["word"] for x in described_text])}")

    sympy_expr = text_to_sympy(described_text)
    print(f"[symbolic expression] {sympy_expr}")
    print(f"[evaluation] {round(sympy_expr.evalf(), 3)}")
