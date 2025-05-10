import argparse
import os
import sys
import openai

# --------------
# Configuration
# --------------
CHAT_MODEL = "gpt-4.1"   # Or "gpt-4o", "gpt-4-turbo", …
TTS_MODEL  = "tts-1"            # Or "tts-1-hd"
VOICE      = "alloy"            # Try "nova", "echo", "fable", "onyx", …

# use 'r' for raw String, so that escape characters are not processed
SYSTEM_PROMPT = r"""

You are a mathematical assistant for blind mathematicians that want to understand a LaTeX. 
Rewrite the given LaTeX so that it can be read aloud clearly. 
Aim for concise, unambiguous phrasing. expand symbols.

Example: '\int_{0}^{\pi} \sin x \, dx': 'the integral from zero to pi of sine x dx'
Example: 'a + (b*c))': 'alpha, plus b times c'
Example: '\sum_{n=1}^{\infty} \frac{1}{n^2}': 'alpha, plus b times c'
Example: '\begin{bmatrix}1 & 2 \\ 7 & 5\end{bmatrix}': 'matrix with rows one comma two, and seven comma five'
"""



def latex_to_description(latex: str) -> str:
    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": latex},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def description_to_speech(text: str, voice: str) -> bytes:
    """
    Uses OpenAI’s TTS endpoint to generate spoken audio.
    Returns raw bytes ready to write to a file.
    """
    resp = openai.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
    )
    return resp.read()


def latex_to_speech(
    latex: str,
    output_path: str = "output.mp3",
    voice: str = VOICE,
):
    description = latex_to_description(latex)
    audio_bytes = description_to_speech(description, voice)

    with open(output_path, "wb") as fh:
        fh.write(audio_bytes)

    print(f"✅ Saved: {output_path}")
    print(f"🗣️  Spoken description:\n{description}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert LaTeX math to spoken audio.")
    p.add_argument("latex", help="LaTeX string (quote or escape backslashes)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    try:
        latex_to_speech(args.latex)
    except openai.OpenAIError as exc:
        sys.exit(f"❌ OpenAI API error: {exc}")


if __name__ == "__main__":
    main()