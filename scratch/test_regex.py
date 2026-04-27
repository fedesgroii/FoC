import re

def test_regex(regex, text):
    try:
        res = re.sub(regex, "MATCH", text)
        print(f"Regex: {regex} | Text: {text} | Result: {res}")
    except Exception as e:
        print(f"Regex: {regex} | Text: {text} | Error: {e}")

# Regex in utils.py
test_regex(r"x_(?:\{(\d+)\}|\s*(\d+))", "x_{1}")
test_regex(r"x_(?:\{(\d+)\}|\s*(\d+))", "x_1")

# Regex in user snippet
test_regex(r"x\_(?:{(\d+)}|\s*(\d+))", "x_{1}")
test_regex(r"x\_(?:{(\d+)}|\s*(\d+))", "x_1")
