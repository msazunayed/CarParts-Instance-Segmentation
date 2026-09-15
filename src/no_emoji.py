import re
import sys

# Match common emoji, symbols, and variation selectors.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "\U0000200D"
    "]+"
)


class _EmojiStrippingStream:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text):
        return self._wrapped.write(_EMOJI_RE.sub("", text))

    def __getattr__(self, name):
        # Pass through methods that belong to the original stream.
        return getattr(self._wrapped, name)


def strip_emojis_streams():
    """Wrap sys.stdout and sys.stderr so no emoji reach the terminal."""
    if not isinstance(sys.stdout, _EmojiStrippingStream):
        sys.stdout = _EmojiStrippingStream(sys.stdout)
    if not isinstance(sys.stderr, _EmojiStrippingStream):
        sys.stderr = _EmojiStrippingStream(sys.stderr)
