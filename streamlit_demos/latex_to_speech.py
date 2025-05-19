import argparse
import os
import sys
import openai
import streamlit as st

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


def run_streamlit_app():
    """Interactive Streamlit UI that converts a LaTeX string to spoken audio."""
    st.set_page_config(layout="wide")

    st.title("📝 LaTeX → Speech Converter")


    col_input, col_result = st.columns([1, 2]) # Layout

    with col_input: # Input Column
        st.subheader("LaTeX Input")
        latex_input = st.text_area("Paste LaTeX here:", height=200)

        voice_options = ["alloy", "nova", "echo", "fable", "onyx"]
        voice_default_idx = voice_options.index(VOICE) if VOICE in voice_options else 0
        voice_choice = st.selectbox("Voice", voice_options, index=voice_default_idx)

        generate_btn = st.button("🔊 Generate Speech", type="primary")

    with col_result: # Result Column
        if generate_btn:
            if not latex_input.strip():
                st.error("Please provide a non‑empty LaTeX expression.")
                return

            try:
                with st.spinner("Generating description and speech …"):
                    description_text = latex_to_description(latex_input)
                    audio_bytes = description_to_speech(description_text, voice_choice)

                st.success("Generation complete!")

                st.markdown("**Spoken Description (text):**")
                st.text_area("Description", description_text, height=150)

                st.markdown("**Audio Playback:**")
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button("💾 Download MP3", data=audio_bytes, file_name="latex_description.mp3", mime="audio/mpeg")

            except openai.OpenAIError as e:
                st.error(f"OpenAI API error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        else:
            st.info("After you click *Generate Speech*, the description and audio will appear here.")


if __name__ == "__main__":
    run_streamlit_app()