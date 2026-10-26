"""
ingest package
================

The two data channels that turn raw media into text evidence:

- image.py   Vision channel (preprocess -> VLM -> description + visible text)
- voice.py   Voice channel (energy VAD/endpointing over a PCM stream -> STT)
- errors.py  Input-validation errors shared by both channels
"""
