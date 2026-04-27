"""
routes/utils.py
Utility condivise tra i moduli di routing.
"""
import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

# ---------------------------------------------------------------------------
# Transformations di parsing SymPy (usate in tutti i route che accettano
# espressioni matematiche dall'utente).
# ---------------------------------------------------------------------------
transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

# Versione senza convert_xor / function_exponentiation (alcuni route la usano)
transformations_basic = standard_transformations + (
    implicit_multiplication_application,
)


# ---------------------------------------------------------------------------
# Sostituzione pedici  x_1, x_{2} → x1, x2
# Usata da: linearizzazione.py
# ---------------------------------------------------------------------------
def sostituisci_pedici(equation_str: str) -> str:
    """
    Converte vari formati di pedici e apici in formato compatibile SymPy.
    Esempi: x_1, x_{2}, x_ 1, x₁, x², x³
    """
    # Mappa pedici Unicode a cifre normali
    unicode_subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    # Mappa apici Unicode a (cifra) per essere poi convertiti in **cifra
    unicode_superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    
    s = equation_str.translate(unicode_subscripts)
    
    # Gestione apici Unicode: trasformiamo x³ in x**3
    for char in "⁰¹²³⁴⁵⁶⁷⁸⁹":
        if char in s:
            digit = char.translate(unicode_superscripts)
            s = s.replace(char, f"**{digit}")

    def replacer(match):
        index = match.group(1) or match.group(2)
        return f"x{index}"
    
    # Regex migliorata per gestire x_1, x_{1}, x_ 1
    return re.sub(r"x\_(?:\{(\d+)\}|\s*(\d+))", replacer, s)


# ---------------------------------------------------------------------------
# Formattazione LaTeX dei valori nei punti di equilibrio
# Usata da: linearizzazione.py
# Converte Float numerici (es. 3.14159…) in costanti simboliche (π, e)
# oppure li arrotonda a 5 cifre significative per la visualizzazione.
# NON deve essere usata per calcoli, solo per output.
# ---------------------------------------------------------------------------
def formatta_x_e(v) -> str:
    """
    Restituisce la stringa LaTeX per un valore di un punto di equilibrio.

    - Float che coincidono con costanti note (π, e) vengono sostituiti.
    - Altri Float vengono arrotondati a 5 decimali; se il decimale è .0
      vengono restituiti come interi.
    - Espressioni composte vengono traversate nodo per nodo.
    """
    from sympy import Float, Integer, nsimplify, simplify, latex

    def _simplify_node(node):
        if isinstance(node, Float):
            simp = nsimplify(node, constants=[sp.pi, sp.E], tolerance=1e-10)
            if isinstance(simp, Float):
                rounded = round(float(simp), 5)
                return Integer(int(rounded)) if rounded == int(rounded) else Float(rounded)
            return simp
        return node

    try:
        # Caso base: non è un'espressione SymPy traversabile
        if not hasattr(v, "replace"):
            if isinstance(v, (Float, float)):
                return latex(_simplify_node(sp.sympify(v)))
            return latex(v)

        v_simp = v.replace(lambda node: isinstance(node, Float), _simplify_node)
        v_simp = simplify(v_simp)
        result = latex(v_simp)
        return "0" if result == "0.0" else result

    except Exception:
        return sp.latex(v)
