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


def parse_frazioni_complete(expr_str: str) -> str:
    """
    Preprocessing delle frazioni per gestire denominatori composti senza parentesi.
    Esempio: 1/x2+1 -> 1/(x2+1)
    Esempio: 1/x + 1/y -> 1/x + 1/y (resta invariato)
    
    Regola: il denominatore di una frazione '/' si estende fino alla fine del blocco
    additivo. Si ferma se incontra un'altra frazione allo stesso livello di parentesi.
    Se il denominatore inizia già con una parentesi, si ferma alla chiusura della stessa.
    """
    if '/' not in expr_str:
        return expr_str

    res = ""
    i = 0
    n = len(expr_str)
    
    while i < n:
        if expr_str[i] == '/':
            # Trovata una frazione. Identifichiamo il denominatore.
            j = i + 1
            # Saltiamo eventuali spazi
            while j < n and expr_str[j].isspace():
                j += 1
            
            if j < n and expr_str[j] == '(':
                # Se il denominatore inizia con '(', cerchiamo solo la sua chiusura
                start_den = j
                level = 1
                j += 1
                while j < n and level > 0:
                    if expr_str[j] == '(': level += 1
                    elif expr_str[j] == ')': level -= 1
                    j += 1
                # Il denominatore è già completo e parentesizzato
                den = expr_str[i+1:j]
                res += f"/{den}"
            else:
                # Altrimenti applichiamo la logica greedy
                start_den = i + 1
                j = start_den
                level = 0
                while j < n:
                    char = expr_str[j]
                    if char == '(': level += 1
                    elif char == ')':
                        if level == 0: break
                        level -= 1
                    elif char == '/' and level == 0:
                        break
                    elif char in ('+', '-') and level == 0:
                        # Regola degli spazi: se l'operatore è preceduto da uno spazio,
                        # lo consideriamo un separatore di termini e non parte del denominatore.
                        if j > start_den and expr_str[j-1].isspace():
                            break
                            
                        # Verifica greedy: se dopo c'è un'altra frazione, questo è un separatore.
                        has_fraction_after = False
                        next_level = 0
                        for k in range(j + 1, n):
                            next_char = expr_str[k]
                            if next_char == '(': next_level += 1
                            elif next_char == ')':
                                if next_level == 0: break
                                next_level -= 1
                            elif next_char == '/' and next_level == 0:
                                has_fraction_after = True
                                break
                            elif next_char in ('+', '-') and next_level == 0:
                                break
                        if has_fraction_after:
                            break
                    j += 1
                
                den = expr_str[start_den:j]
                den_stripped = den.strip()
                if ('+' in den or '-' in den) and not (den_stripped.startswith('(') and den_stripped.endswith(')')):
                    res += f"/({den})"
                else:
                    res += f"/{den}"
            i = j
        else:
            res += expr_str[i]
            i += 1
            
    return res
