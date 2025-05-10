from sympy.parsing.sympy_parser import parse_expr
import openai
import os
from typing import List, Dict
import sympy as sp
import streamlit as st
import tempfile
from audio_recorder_streamlit import audio_recorder

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


def get_results_for_streamlit(path: str) -> Dict[str, str]:
    transcription = transcribe(path)
    plain_text = " ".join([w["word"] for w in transcription])
    print(f"[raw transcription] {plain_text}")

    sympy_expr = text_to_sympy(plain_text)
    print(f"[sympy expression] {sympy_expr}")

    mathml_result = sp.printing.mathml(sympy_expr) # use printer='presentation' if you want presentation mathml
    return {
        "transcription": plain_text,
        "sympy_expr_str": str(sympy_expr),
        "mathml": mathml_result
    }


def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("🎙️ Voice to MathML Converter")
    st.markdown(
        "Record your mathematical expression using your microphone. "
        "The system will transcribe it, convert it to a SymPy expression, and then render it as MathML."
    )

    col_mic, col_results = st.columns([1, 2])

    with col_mic:
        st.subheader("Record Audio Input")
        audio_bytes = audio_recorder( # The audio_recorder widget for voice input
            text="Click the icon to record:",
            recording_color="#e84343",  # red color means recording
            neutral_color="#008E00",  # green color means not recording
            icon_size="3x",  # Larger icon
            pause_threshold=2.0,  # Shorter silence to stop
        )

    with col_results:
        st.subheader("Output")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")  # Display the recorded audio

            # Temporary file to store audio bytes for processing
            tmp_audio_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
                    tmp_audio_file.write(audio_bytes)
                    tmp_audio_path = tmp_audio_file.name

                with st.spinner("Processing your voice input... Please wait."):
                    results = get_results_for_streamlit(tmp_audio_path)

                st.success("Processing complete!")
                st.markdown("---")  # Visual separator

                # Displaying the results
                st.markdown(f"**Transcribed Text:**")
                st.text_area("Transcription", results["transcription"], height=75, key="transcription_display")

                st.markdown(f"**SymPy Expression (Python Code):**")
                st.code(results["sympy_expr_str"], language="python")

                st.markdown(f"**MathML:**")
                st.code(results['mathml'], language="xml")

            except InvalidSympyException as e:
                st.error(f"SymPy Conversion Error: {e}. Please ensure your spoken math is clear or try rephrasing.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.error("Details: Check your microphone, internet connection, and OpenAI API key / quota.")
            finally:
                if tmp_audio_path and os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)  # Clean up the temp file
        else:
            st.info("After recording, the results will be displayed here.")


if __name__ == "__main__":
    # This replaces the original argparse logic
    run_streamlit_app()
