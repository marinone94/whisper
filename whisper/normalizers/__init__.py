try:
    from .basic import BasicTextNormalizer
    from .english import EnglishTextNormalizer
except ImportError:
    from whisper.normalizers.basic import BasicTextNormalizer
    from whisper.normalizers.english import EnglishTextNormalizer