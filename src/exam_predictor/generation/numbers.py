"""Numbers: Python is the calculator, because language models often get arithmetic wrong."""
import math
import re

DATA_SHEET = {"g": 9.81, "c": 3.00e8, "e": 1.60e-19, "u": 1.66e-27, "m_p": 1.67e-27, "m_e": 9.11e-31, "N_A": 6.02e23,
              "R": 8.31, "k": 1.38e-23, "G": 6.67e-11, "eps0": 8.85e-12, "h": 6.63e-34, "sigma": 5.67e-8}
SAFE = {**DATA_SHEET, "pi": math.pi, "abs": abs, **{f: getattr(math, f) for f in
        ["sqrt", "exp", "log", "log10", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "radians", "degrees"]}}
EVERYDAY = {0.5, 1.5, 0.25, 0.693, 24, 60, 100, 180, 273, 273.15, 360, 365, 3600}   # numbers a calculation may always use
NUMBER = re.compile(r"[-+−–]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*[×x]\s*10\s*\^\s*[-+−–]?\d+)?")
STATED = re.compile(r"(?:\bis|\bof|\bbe|=)\s+(?:about\s+|approximately\s+)?(" + NUMBER.pattern + ")")


def to_number(text):
    """Read a number written as text: '19.6' -> 19.6, '2.5 × 10^3' -> 2500.0."""
    text = text.replace("−", "-").replace("–", "-")
    m = re.match(r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*[×x]\s*10\s*\^\s*([-+]?\d+))?", text)
    return float(m[1]) * 10.0 ** int(m[2] or 0) if m else None


def numbers_in(text):
    """All numbers in a text, e.g. 'is 12 000 K and 2.5 × 10^3 Pa' -> [12000.0, 2500.0]."""
    text = re.sub(r"(\d) (?=\d{3}\b)", r"\1", text)                 # Cambridge writes 12 000 for 12000
    return [to_number(n) for n in NUMBER.findall(text)]


def nice(x):
    """A number written the Cambridge way, to 3 significant figures: 0.0125, 47.6, 8.94 × 10^6."""
    if x != 0 and not 0.01 <= abs(x) < 10000:
        mantissa, power = f"{x:.2e}".split("e")
        return f"{mantissa} × 10^{int(power)}"
    return f"{float(f'{x:.3g}'):g}"


def same_value(a, b):
    """True if a and b agree within 5%, also when one has a unit prefix (75 mA and 0.075 A)."""
    return any(math.isclose(a * 1000.0 ** k, b, rel_tol=0.05) for k in range(-4, 5))


def clean_expression(expression):
    """Remove common slips from a model's Python expression: 'import math; math.sqrt(2^2)' -> 'sqrt(2**2)'."""
    return re.sub(r"import \w+;?|\b(math|np|numpy)\.", "", expression).replace("^", "**").strip()


def run(expression, names=None):
    """Calculate a Python expression safely, with the data values (and earlier answers or x)."""
    return float(eval(clean_expression(expression), {"__builtins__": {}}, {**SAFE, **(names or {})}))


def pretty(expression):
    """'(120*180)/(2.5*ans_b_i)' -> '(120 × 180)/(2.5 × (b)(i))', used as the substitution step of a mark scheme."""
    expression = re.sub(r"ans_(\w+)", lambda m: "".join(f"({t})" for t in m[1].split("_")), clean_expression(expression))
    return re.sub(r"\s+", " ", expression.replace("**", "^").replace("*", " × ")).strip()


def unknown_numbers(calculation, given):
    """Numbers in a calculation that are not in the question's data (or the data sheet): a sign of a made-up value."""
    found = [float(x) for x in re.findall(r"(?<![\w.])\d+\.?\d*(?:[eE][-+]?\d+)?", calculation)]
    return [x for x in found if x and not (x == int(x) and x <= 12) and x not in EVERYDAY
            and not math.log10(x).is_integer() and not any(same_value(x, g) for g in given)]
