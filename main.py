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


def text_to_latex(natural_text: str) -> str:
    examples = [
        {
            "text": "the integral from zero to pi of sine x d x",
            "latex": r"\int_{0}^{\pi} \sin x \, dx",
        },
        {
            "text": "the sum from n equals one to infinity of the fraction one over n squared",
            "latex": r"\sum_{n = 1}^{\infty} \frac{1}{n^{2}}",
        },
        {
            "text": "alpha squared plus beta sub zero all over gamma",
            "latex": r"\frac{\alpha^{2} + \beta_{0}}{\gamma}",
        },
    ]
    examples_text = "\n\n".join(
        f"Text: {ex['text']}\nLaTeX: {ex['latex']}" for ex in examples
    )

    prompt = f"""Convert the following natural‑language description of a mathematical expression
    into  **pure LaTeX** (math mode) without surrounding dollar signs or backticks..
    Use the word level timestamps to infer which parts to group together in the expression.

    Here are some examples:

    {examples_text}

    Text: {natural_text}
    Expression:"""

    response = openai.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that converts spoken or written mathematics "
                    "into high‑quality LaTeX code."
                ),
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0,
        max_tokens=200,
    )

    return response.choices[0].message.content.strip()



def audio_to_latex(path: str) -> None:
    """End‑to‑end pipeline: audio → text → LaTeX (→ optional evaluation)."""

    words = transcribe(path)
    natural_text = " ".join(w["word"] for w in words)
    print(f"[raw transcription] {natural_text}")

    latex_expr = text_to_latex(natural_text)
    print(f"[LaTeX expression] {latex_expr}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to an audio file (wav/mp3/ogg…)")
    args = ap.parse_args()

    audio_to_latex(path=args.audio)
