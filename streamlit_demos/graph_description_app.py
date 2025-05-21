from __future__ import annotations

import os
import textwrap

import openai
import requests
import streamlit as st


openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14")


SYSTEM_PROMPT = textwrap.dedent(
"""
You are an assistive-technology expert creating an *accessible, concise and precise* description
of a mathematical function graph for blind university students.  
Carefully analyse the image and answer with the following six sections **in exactly this order**, 
each starting with the section title followed by a colon. Leave the body empty if nothing relevant is found.

1. metadata:  
2. coordinate system:  
3. environment:
4. function names:  
5. description of the functions:  
6. other notable characteristics:

### Content rules for every section

**metadata** – State the type of graphic (e.g. “function plot”), the title or caption if visible, and the number of depicted functions.

**coordinate system** – Describe the visible axes (range, units, scale, quadrants shown), tick labels, grid lines, origin position and any axis arrows.

**environment** – Describe additional graphic elements *not* belonging to the functions themselves (e.g. shaded areas, ε–δ rectangles, legends, highlighted segments, reference lines, texts or symbols). 

**function names** – List every function in the legend order; if the image lacks explicit names, invent neutral identifiers (f, g, …). Add the visual cue that helps distinguish them (colour, line style, marker).

**description of the functions** –  
* For **each** function give two parts **in the following pattern**  
  - *Continuous text*: Describe the course of the curve **from left to right**, mentioning monotonicity, curvature, intersections with axes, maxima/minima, asymptotes, points where several curves meet, etc. End this paragraph with a full stop.  
  - *Coordinate list*: Start a new line `function <name> coordinates:` and list the key points as “– (x|y)” items in ascending x-order (start/end, intercepts, extrema, inflection points, intersections with other curves). Approximate if necessary and say “approximately”.

**other notable characteristics** – Mention global properties such as symmetry (even/odd, axis, point), periodicity, identical intercepts shared by all functions, unusual styling, or anything still important that has not been covered above. 

### Style & wording

* Use short, factual sentences; avoid formulas unless printed in the image.  
* Express numbers in standard decimal notation with “|” as coordinate separator, e.g. “(−0.4|7.5)”.  
* When ranges are clear, prefer exact values; otherwise prefix with “approximately”.  
* Keep each section logically self-contained; do **not** duplicate information across sections.  
* Do **not** include headings or text beyond the six prescribed sections.

Return only the six sections, nothing else.
Return in plain text, w
"""
)


def call_openai_vision(url: str):
    client = openai.Client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
        ],
    )
    return response.choices[0].message.content



st.set_page_config(page_title="Graph Description Generator", layout="wide")
st.title("Graph Description Generator")

url: str = st.text_input("Graph image URL", placeholder="https://example.com/graph.png")

if url:
    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown("**Input graph**")
        try:
            image_data = requests.get(url, timeout=5).content
            st.image(image_data, use_container_width=True)
        except Exception as ex:
            st.error(f"Error loading image: {ex}")

    with cols[1]:
        st.markdown("**Generated description**")
        if st.button("Generate", type="primary"):
            try:
                description = call_openai_vision(url)
            except Exception as ex:
                st.error(f"OpenAI request failed: {ex}")
            else:
                st.markdown(description)
