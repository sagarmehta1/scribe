"""Scribe — a free, local alternative to Otter.ai."""

__version__ = "0.1.0"

# Use the operating system's certificate store for TLS verification. This lets
# outbound HTTPS (model downloads, Anthropic/OpenAI) work behind corporate or
# antivirus TLS interception that injects its own root CA, which certifi's
# bundle doesn't know about. No-op if truststore isn't installed.
try:
    import truststore as _truststore
    _truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass
