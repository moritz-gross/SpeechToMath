# Speech-To-Math

## Current pipeline:
- Transcribe audio with OpenAI Whisper, including word-level timestamps
- post-process with gpt-4.1, using some examples for in context learning
- parse result to SymPy expression


### Potential TODOs
- chain-of-thought for better performance, e.g. using OpenAI o3
- retries (using error message) in case parsing to SymPy doesn't work
- local models?? Qwen3 maybe??
- add unit tests

### Trivia
- Turn m4a into mp3 for wider support:
  - ```for f in *.m4a; do ffmpeg -i "$f" -codec:a libmp3lame -qscale:a 2 "${f%.m4a}.mp3"; done```