"""
providers package
===================

Everything that talks to a model lives behind small interfaces defined in
base.py, so the pipeline never imports a vendor SDK:

- base.py           result types, Protocols, ProviderUnavailableError
- fakes.py          deterministic providers for DEMO_MODE and tests
- openai_compat.py  adapters for OpenAI-compatible APIs (OpenRouter, OpenAI)
- gateway.py        resilience layer: timeout, retry, fallback, counters
- factory.py        builds the right provider set from Settings
"""
