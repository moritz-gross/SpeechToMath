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


SYSTEM_PROMPT = (
    "You are a mathematical narrator speaking to a general audience. "
    "Rewrite the given LaTeX equation so that it can be read aloud clearly. "
    "Aim for concise, unambiguous phrasing; expand symbols (e.g. "
    "'∫₀^π sin x dx' → 'the integral from zero to pi of sine x dx')."
)


def latex_to_description(latex: str) -> str:
    """
    Turn a LaTeX math string into a human‑readable sentence.
    """
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