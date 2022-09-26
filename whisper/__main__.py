try:
    from .transcribe import cli
except ImportError:
    from whisper.transcribe import cli


cli()
