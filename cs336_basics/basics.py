def demonstrate_utf_encodings():
    """
    Demonstrates the differences between UTF-8, UTF-16, and UTF-32 encodings
    and explains why UTF-8 is preferred for tokenizer training.
    """

    # Test strings with different character types
    test_strings = [
        "Hello World",  # ASCII characters
        "Hello 世界",  # Mixed ASCII and Unicode
        "こんにちは",  # Japanese characters
        "Привет мир",  # Cyrillic characters
        "🌟🚀💻",  # Emojis
        "Hello\nWorld\tTab",  # Control characters
    ]

    print("UTF Encoding Comparison Example")
    print("=" * 50)

    for i, text in enumerate(test_strings, 1):
        print(f"\n{i}. Input: '{text}'")
        print(f"   Length: {len(text)} characters")

        # Encode to different UTF formats
        utf8_bytes = text.encode("utf-8")
        utf16_bytes = text.encode("utf-16")
        utf32_bytes = text.encode("utf-32")

        print(f"   UTF-8:  {len(utf8_bytes)} bytes - {utf8_bytes}")
        print(f"   UTF-16: {len(utf16_bytes)} bytes - {utf16_bytes}")
        print(f"   UTF-32: {len(utf32_bytes)} bytes - {utf32_bytes}")

    print("\n" + "=" * 50)
    print("Why UTF-8 is preferred for tokenizer training:")
    print("1. Space efficiency: UTF-8 uses 1 byte for ASCII characters (most common)")
    print("2. Backward compatibility: UTF-8 is a superset of ASCII")
    print("3. Subword tokenization: UTF-8 allows byte-level tokenization")
    print("4. Memory efficiency: Smaller vocabulary size needed")
    print("5. Processing speed: Faster encoding/decoding for most text")


def analyze_encoding_efficiency():
    """
    Analyzes the efficiency of different UTF encodings for common text.
    """

    # Sample text with typical distribution
    sample_text = """
    Hello World! This is a sample text with some ASCII characters.
    We also have some Unicode: 你好世界, привет, こんにちは, 🌟🚀
    And some numbers: 1234567890
    """

    print("\nEncoding Efficiency Analysis")
    print("=" * 40)
    print(f"Sample text length: {len(sample_text)} characters")

    utf8_size = len(sample_text.encode("utf-8"))
    utf16_size = len(sample_text.encode("utf-16"))
    utf32_size = len(sample_text.encode("utf-32"))

    print(f"UTF-8 size:  {utf8_size} bytes")
    print(f"UTF-16 size: {utf16_size} bytes")
    print(f"UTF-32 size: {utf32_size} bytes")

    print(f"\nUTF-8 is {(utf16_size/utf8_size):.1f}x smaller than UTF-16")
    print(f"UTF-8 is {(utf32_size/utf8_size):.1f}x smaller than UTF-32")


def demonstrate_incorrect_utf8_decoding():
    """
    Demonstrates why the given UTF-8 decoding function is incorrect.
    The function incorrectly processes UTF-8 bytes one at a time instead of
    properly handling multi-byte sequences.
    """

    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        """Incorrect UTF-8 decoding function that processes bytes individually."""
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])

    def decode_utf8_bytes_to_str_correct(bytestring: bytes):
        """Correct UTF-8 decoding function."""
        return bytestring.decode("utf-8")

    print("\nIncorrect UTF-8 Decoding Example")
    print("=" * 50)

    # Test cases
    test_cases = [
        ("hello", "ASCII characters (works correctly)"),
        ("世界", "Chinese characters (fails)"),
        ("こんにちは", "Japanese characters (fails)"),
        ("🌟", "Emoji (fails)"),
        ("Hello 世界", "Mixed ASCII and Unicode (partially fails)"),
    ]

    for text, description in test_cases:
        print(f"\nInput text: '{text}' ({description})")
        print(f"Text length: {len(text)} characters")

        # Encode to UTF-8 bytes
        utf8_bytes = text.encode("utf-8")
        print(f"UTF-8 bytes: {utf8_bytes} (length: {len(utf8_bytes)} bytes)")

        # Try both decoding methods
        try:
            wrong_result = decode_utf8_bytes_to_str_wrong(utf8_bytes)
            print(f"Wrong method result: '{wrong_result}'")
        except UnicodeDecodeError as e:
            print(f"Wrong method error: {e}")

        try:
            correct_result = decode_utf8_bytes_to_str_correct(utf8_bytes)
            print(f"Correct method result: '{correct_result}'")
        except UnicodeDecodeError as e:
            print(f"Correct method error: {e}")

        # Check if results match
        try:
            wrong_result = decode_utf8_bytes_to_str_wrong(utf8_bytes)
            correct_result = decode_utf8_bytes_to_str_correct(utf8_bytes)
            if wrong_result == correct_result:
                print("✓ Results match (function works for this input)")
            else:
                print("✗ Results differ (function fails for this input)")
        except UnicodeDecodeError:
            print("✗ Wrong method fails with UnicodeDecodeError")

    print("\n" + "=" * 50)
    print("Why the function is incorrect:")
    print("The function processes UTF-8 bytes individually, but UTF-8 uses")
    print("variable-length encoding where characters can span multiple bytes.")
    print("For example, '世界' in UTF-8 is 6 bytes, but the function tries")
    print("to decode each byte separately, which fails for multi-byte sequences.")


def demonstrate_invalid_utf8_sequences():
    """
    Demonstrates invalid UTF-8 byte sequences that cannot be decoded
    to any Unicode character.
    """

    print("\nInvalid UTF-8 Byte Sequences")
    print("=" * 50)

    # Invalid UTF-8 byte sequences
    invalid_sequences = [
        (b"\xff\xff", "Two 0xFF bytes - invalid continuation bytes"),
        (b"\xc0\x80", "Overlong encoding of null character"),
        (b"\xe0\x80\x80", "Overlong encoding of null character (3 bytes)"),
        (b"\xf0\x80\x80\x80", "Overlong encoding of null character (4 bytes)"),
        (
            b"\xc1\xbf",
            "Invalid 2-byte sequence (should be 0xC0-0xDF followed by 0x80-0xBF)",
        ),
        (b"\xe0\x9f\xbf", "Invalid 3-byte sequence (second byte out of range)"),
        (b"\xf0\x8f\xbf\xbf", "Invalid 4-byte sequence (second byte out of range)"),
        (b"\x80\x80", "Two continuation bytes without leading byte"),
        (b"\xc0\x7f", "Leading byte followed by non-continuation byte"),
    ]

    for i, (byte_seq, description) in enumerate(invalid_sequences, 1):
        print(f"\n{i}. {description}")
        print(f"   Byte sequence: {byte_seq}")
        print(f"   Hex representation: {byte_seq.hex()}")

        try:
            result = byte_seq.decode("utf-8")
            print(f"   Decode result: '{result}' (unexpected - should fail)")
        except UnicodeDecodeError as e:
            print(f"   Decode error: {e}")
            print(f"   ✓ Correctly rejected as invalid UTF-8")

    print("\n" + "=" * 50)
    print("Why these sequences are invalid:")
    print("1. 0xFF is never a valid UTF-8 byte")
    print("2. Overlong encodings use more bytes than necessary")
    print("3. Continuation bytes (0x80-0xBF) must follow leading bytes")
    print("4. Leading bytes must be followed by correct number of continuation bytes")
    print("5. Byte ranges must be within valid UTF-8 encoding rules")


if __name__ == "__main__":
    demonstrate_utf_encodings()
    analyze_encoding_efficiency()
    demonstrate_incorrect_utf8_decoding()
    demonstrate_invalid_utf8_sequences()
