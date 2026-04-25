"""
routes/equazioni_differenziali.py
Risolutore completo di Equazioni Differenziali Ordinarie (ODE).
Supporta ODE del primo e secondo ordine con risoluzione passo-passo in LaTeX.
"""

from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import (
    symbols, Function, Eq, dsolve, Derivative, exp, simplify,
    cos, sin, tan, ln, log, latex, expand, solve, I, pi, oo,
    classify_ode, sqrt, Rational, Abs, sign, integrate, diff,
    trigsimp, powsimp, cancel, factor, collect, apart
)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
import re
import traceback

equazioni_differenziali_bp = Blueprint("equazioni_differenziali", __name__)

# ── Trasformazioni per il parser ──────────────────────────────────────────
_transformations = standard_transformations + (implicit_multiplication_application, convert_xor,)

# ── Variabili simboliche globali ──────────────────────────────────────────
x = symbols('x', real=True)
y_func = Function('y')
y = y_func(x)
C1, C2, C3 = symbols('C1 C2 C3')


# ══════════════════════════════════════════════════════════════════════════
#  ENDPOINT API
# ══════════════════════════════════════════════════════════════════════════
@equazioni_differenziali_bp.route('/api/equazioni_differenziali', methods=['POST'])
def api_equazione_differenziale():
    data = request.get_json()
    equazione = data.get("equazione", "").strip()
    x0 = data.get("x0", "").strip()
    y0 = data.get("y0", "").strip()
    dy0 = data.get("dy0", "").strip()

    if not equazione:
        return jsonify({"success": False, "error": "Nessuna equazione differenziale fornita."})

    # Costruisci dizionario condizioni iniziali
    condizioni = {}
    try:
        if x0 and y0:
            condizioni['x0'] = _parse_valore(x0)
            condizioni['y0'] = _parse_valore(y0)
        if x0 and dy0:
            condizioni['x0'] = _parse_valore(x0)
            condizioni['dy0'] = _parse_valore(dy0)
    except Exception as e:
        return jsonify({"success": False, "error": f"Errore nel parsing delle condizioni iniziali: {e}"})

    result = risolvi_equazione_differenziale(equazione, condizioni if condizioni else None)
    return jsonify(result)


def _parse_valore(s):
    """Converte una stringa in un'espressione SymPy."""
    s = s.strip().replace('^', '**')
    local = {'e': sp.E, 'pi': pi, 'inf': oo}
    return parse_expr(s, local_dict=local, transformations=_transformations)


# ══════════════════════════════════════════════════════════════════════════
#  PARSING DELL'INPUT
# ══════════════════════════════════════════════════════════════════════════
def parsifica_input(testo):
    """
    Converte una stringa di equazione differenziale in un'equazione SymPy.
    Formati supportati:
      - y' = x*y
      - y'' + 3y' + 2y = e^x
      - dy/dx = x + y
      - x^2*y'' + x*y' + y = 0
    Restituisce (eq_sympy, ordine, latex_input).
    """
    testo_orig = testo.strip()
    testo = testo_orig

    # Normalizzazione di base
    testo = testo.replace('^', '**')
    testo = testo.replace('·', '*')

    # Gestione dy/dx, d²y/dx², d2y/dx2
    testo = re.sub(r'd\s*²\s*y\s*/\s*d\s*x\s*²', "y''", testo)
    testo = re.sub(r'd\s*2\s*y\s*/\s*d\s*x\s*2', "y''", testo)
    testo = re.sub(r'd\s*y\s*/\s*d\s*x', "y'", testo)

    # Separazione LHS = RHS
    if '=' not in testo:
        raise ValueError("L'equazione deve contenere il segno '='. Esempio: y' = x*y")

    lhs_str, rhs_str = testo.split('=', 1)
    lhs_str = lhs_str.strip()
    rhs_str = rhs_str.strip()
    if not rhs_str:
        rhs_str = '0'

    # Converti la stringa in espressione SymPy
    lhs_expr = _parse_ode_side(lhs_str)
    rhs_expr = _parse_ode_side(rhs_str)

    # Crea l'equazione SymPy: lhs - rhs = 0
    eq = Eq(lhs_expr, rhs_expr)

    # Determina l'ordine
    ordine = _determina_ordine(testo_orig)

    # LaTeX dell'input
    latex_input = latex(eq)

    return eq, ordine, latex_input


def _parse_ode_side(s):
    """Converte una parte (LHS o RHS) in un'espressione SymPy con derivate."""
    s = s.strip()
    if not s:
        return sp.Integer(0)

    # Gestione y''', y'', y' → Derivative
    # Processa in ordine decrescente di derivata
    s = re.sub(r"y\s*'''", '__DER3__', s)
    s = re.sub(r"y\s*''", '__DER2__', s)
    s = re.sub(r"y\s*'", '__DER1__', s)

    # Gestione y^(n)
    s = re.sub(r"y\s*\*\*\s*\(\s*(\d+)\s*\)", lambda m: f'__DER{m.group(1)}__', s)

    # Sostituzione y → y(x) placeholder
    s = re.sub(r'\by\b(?!_)', '__YFUNC__', s)

    local_dict = {
        'x': x, 'e': sp.E, 'pi': pi, 'E': sp.E,
        'sin': sin, 'cos': cos, 'tan': tan,
        'exp': exp, 'ln': ln, 'log': log,
        'sqrt': sqrt, 'abs': Abs,
        '__YFUNC__': y,
        '__DER1__': Derivative(y, x),
        '__DER2__': Derivative(y, x, x),
        '__DER3__': Derivative(y, (x, 3)),
    }

    # Aggiungi moltiplicazione implicita dove necessario
    # Es: "3y'" → "3*y'", "xy" → "x*y"
    s = re.sub(r'(\d)(__DER|__YFUNC)', r'\1*\2', s)
    s = re.sub(r'(__DER\d__|__YFUNC__)(\d)', r'\1*\2', s)
    s = re.sub(r'(\))(__DER|__YFUNC)', r'\1*\2', s)
    s = re.sub(r'(x)(__DER|__YFUNC)', r'\1*\2', s)
    s = re.sub(r'(__DER\d__|__YFUNC__)(x|\()', r'\1*\2', s)
    s = re.sub(r'(\d)(x\b)', r'\1*\2', s)
    s = re.sub(r'(x\b)(\d)', r'\1*\2', s)

    try:
        expr = parse_expr(s, local_dict=local_dict, transformations=_transformations)
    except Exception as e:
        raise ValueError(f"Impossibile interpretare l'espressione '{s}': {e}")

    return expr


def _determina_ordine(testo):
    """Determina l'ordine dell'ODE dalla stringa."""
    if "y'''" in testo or "y***(3)" in testo or "d³y" in testo:
        return 3
    if "y''" in testo or "y**(2)" in testo or "d²y" in testo or "d2y" in testo:
        return 2
    return 1


# ══════════════════════════════════════════════════════════════════════════
#  CLASSIFICAZIONE
# ══════════════════════════════════════════════════════════════════════════

# Mappa le classificazioni SymPy a nomi leggibili in italiano
_TIPO_NOMI = {
    'separable': 'Equazione a variabili separabili',
    '1st_linear': 'Equazione lineare del primo ordine',
    'Bernoulli': 'Equazione di Bernoulli',
    '1st_homogeneous_coeff': 'Equazione omogenea',
    '1st_exact': 'Equazione differenziale esatta',
    'nth_linear_constant_coeff_homogeneous': 'Equazione lineare a coefficienti costanti omogenea',
    'nth_linear_constant_coeff_undetermined_coefficients': 'Equazione lineare a coefficienti costanti non omogenea',
    'nth_linear_constant_coeff_variation_of_parameters': 'Equazione lineare a coefficienti costanti (variazione parametri)',
    'nth_linear_euler_eq_homogeneous': 'Equazione di Cauchy-Eulero omogenea',
    'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters': 'Equazione di Cauchy-Eulero non omogenea',
}


def classifica_ode(eq):
    """
    Classifica l'ODE usando SymPy e restituisce (tipo_key, nome_italiano, hints_list).
    """
    try:
        hints = classify_ode(eq, y)
        if not hints:
            return 'generico', 'Equazione differenziale generica', []
    except Exception:
        return 'generico', 'Equazione differenziale generica', []

    hints_list = list(hints)

    # Cerca la prima corrispondenza nella nostra mappa
    for h in hints_list:
        h_clean = h.replace('_Integral', '')
        for key, nome in _TIPO_NOMI.items():
            if key in h_clean:
                return key, nome, hints_list

    return hints_list[0] if hints_list else 'generico', 'Equazione differenziale', hints_list


# ══════════════════════════════════════════════════════════════════════════
#  GENERAZIONE PASSAGGI RISOLUTIVI
# ══════════════════════════════════════════════════════════════════════════
def genera_passaggi(eq, tipo_key, ordine, soluzione_generale, soluzione_particolare=None, condizioni=None):
    """
    Genera la lista di passaggi LaTeX per la risoluzione.
    Ogni passaggio è {"title": ..., "content": ...}.
    """
    steps = []

    # ── Passaggio: Classificazione ────────────────────────────────────
    _, nome_tipo, _ = classifica_ode(eq)
    steps.append({
        "title": "Classificazione dell'equazione",
        "content": rf"\text{{{nome_tipo} di ordine {ordine}}}"
    })

    # ── Passaggi specifici per tipo ───────────────────────────────────
    try:
        eq_lhs = eq.lhs
        eq_rhs = eq.rhs

        if 'separable' in tipo_key:
            steps += _passaggi_separabile(eq_lhs, eq_rhs)
        elif '1st_linear' in tipo_key:
            steps += _passaggi_lineare_primo(eq_lhs, eq_rhs)
        elif 'Bernoulli' in tipo_key:
            steps += _passaggi_bernoulli(eq_lhs, eq_rhs)
        elif 'nth_linear_constant_coeff' in tipo_key:
            steps += _passaggi_coeff_costanti(eq, ordine)
        elif 'euler' in tipo_key:
            steps += _passaggi_eulero(eq, ordine)
        elif '1st_exact' in tipo_key:
            steps += _passaggi_esatta(eq_lhs, eq_rhs)
        elif '1st_homogeneous' in tipo_key:
            steps += _passaggi_omogenea(eq_lhs, eq_rhs)
        else:
            steps.append({
                "title": "Metodo risolutivo",
                "content": r"\text{Risoluzione tramite metodi simbolici generali (SymPy dsolve)}"
            })
    except Exception:
        steps.append({
            "title": "Metodo risolutivo",
            "content": r"\text{Risoluzione automatica tramite SymPy}"
        })

    # ── Soluzione Generale ────────────────────────────────────────────
    sol_latex = latex(simplify(soluzione_generale))
    steps.append({
        "title": "Soluzione generale",
        "content": rf"y(x) = {sol_latex}"
    })

    # ── Soluzione Particolare (Cauchy) ────────────────────────────────
    if soluzione_particolare is not None and condizioni:
        cond_parts = []
        if 'y0' in condizioni:
            cond_parts.append(rf"y({latex(condizioni['x0'])}) = {latex(condizioni['y0'])}")
        if 'dy0' in condizioni:
            cond_parts.append(rf"y'({latex(condizioni['x0'])}) = {latex(condizioni['dy0'])}")
        cond_str = r", \quad ".join(cond_parts)

        steps.append({
            "title": "Applicazione delle condizioni iniziali",
            "content": cond_str
        })

        sol_part_latex = latex(simplify(soluzione_particolare))
        steps.append({
            "title": "Soluzione particolare (problema di Cauchy)",
            "content": rf"\boxed{{y(x) = {sol_part_latex}}}"
        })

    return steps


# ── Passaggi: Variabili Separabili ────────────────────────────────────
def _passaggi_separabile(lhs, rhs):
    steps = []
    steps.append({
        "title": "Metodo: Separazione delle variabili",
        "content": r"\text{Separiamo le variabili } x \text{ e } y \text{ ai due lati dell'equazione}"
    })
    steps.append({
        "title": "Forma separata",
        "content": r"\frac{dy}{g(y)} = f(x)\,dx"
    })
    steps.append({
        "title": "Integrazione di entrambi i membri",
        "content": r"\int \frac{dy}{g(y)} = \int f(x)\,dx + C"
    })
    return steps


# ── Passaggi: Lineare Primo Ordine ────────────────────────────────────
def _passaggi_lineare_primo(lhs, rhs):
    steps = []
    steps.append({
        "title": "Forma standard",
        "content": r"y' + p(x)\,y = q(x)"
    })
    steps.append({
        "title": "Fattore integrante",
        "content": r"\mu(x) = e^{\int p(x)\,dx}"
    })
    steps.append({
        "title": "Moltiplicazione per il fattore integrante",
        "content": r"\frac{d}{dx}\left[\mu(x)\,y\right] = \mu(x)\,q(x)"
    })
    steps.append({
        "title": "Integrazione",
        "content": r"y = \frac{1}{\mu(x)}\left[\int \mu(x)\,q(x)\,dx + C\right]"
    })
    return steps


# ── Passaggi: Bernoulli ───────────────────────────────────────────────
def _passaggi_bernoulli(lhs, rhs):
    steps = []
    steps.append({
        "title": "Forma di Bernoulli",
        "content": r"y' + p(x)\,y = q(x)\,y^n"
    })
    steps.append({
        "title": "Sostituzione",
        "content": r"v = y^{1-n} \implies v' = (1-n)\,y^{-n}\,y'"
    })
    steps.append({
        "title": "Equazione linearizzata in v",
        "content": r"v' + (1-n)\,p(x)\,v = (1-n)\,q(x)"
    })
    steps.append({
        "title": "Risoluzione dell'equazione lineare in v",
        "content": r"\text{Si risolve l'equazione lineare e si ritorna a } y = v^{1/(1-n)}"
    })
    return steps


# ── Passaggi: Coeff. Costanti ─────────────────────────────────────────
def _passaggi_coeff_costanti(eq, ordine):
    steps = []

    # Prova a estrarre coefficienti dall'equazione
    try:
        # Riporta l'eq in forma lhs = rhs dove lhs ha le derivate
        eq_expr = eq.lhs - eq.rhs
        # Raccogli i coefficienti
        dy2 = eq_expr.coeff(Derivative(y, x, x))
        dy1 = eq_expr.coeff(Derivative(y, x))
        dy0 = eq_expr.coeff(y)

        if ordine >= 2 and (dy2 != 0 or dy1 != 0 or dy0 != 0):
            # Equazione caratteristica
            r = symbols('r')
            if ordine == 2:
                char_poly = dy2 * r**2 + dy1 * r + dy0
            else:
                char_poly = dy1 * r + dy0

            char_poly_s = sp.Poly(char_poly, r) if char_poly.has(r) else char_poly
            steps.append({
                "title": "Equazione caratteristica",
                "content": rf"{latex(char_poly)} = 0"
            })

            # Radici
            radici = solve(char_poly, r)
            radici_str = ", ".join([rf"r = {latex(rad)}" for rad in radici])
            steps.append({
                "title": "Radici dell'equazione caratteristica",
                "content": radici_str
            })

            # Tipo di soluzione basata sulle radici
            if ordine == 2 and len(radici) == 2:
                r1, r2 = radici
                if r1 != r2 and r1.is_real and r2.is_real:
                    steps.append({
                        "title": "Radici reali distinte",
                        "content": rf"y_o(x) = C_1 e^{{{latex(r1)} x}} + C_2 e^{{{latex(r2)} x}}"
                    })
                elif r1 == r2:
                    steps.append({
                        "title": "Radice reale doppia",
                        "content": rf"y_o(x) = (C_1 + C_2 x)\,e^{{{latex(r1)} x}}"
                    })
                elif not r1.is_real:
                    alpha = sp.re(r1)
                    beta = sp.Abs(sp.im(r1))
                    steps.append({
                        "title": "Radici complesse coniugate",
                        "content": rf"\alpha = {latex(alpha)},\; \beta = {latex(beta)}"
                    })
                    steps.append({
                        "title": "Soluzione omogenea",
                        "content": rf"y_o(x) = e^{{{latex(alpha)} x}}\left(C_1 \cos({latex(beta)} x) + C_2 \sin({latex(beta)} x)\right)"
                    })

            # Termine forzante
            # Ricostruisci il termine noto
            termine_noto = eq.rhs
            if termine_noto != 0 and not sp.simplify(termine_noto).is_zero:
                steps.append({
                    "title": "Termine forzante (parte non omogenea)",
                    "content": rf"f(x) = {latex(termine_noto)}"
                })
                steps.append({
                    "title": "Soluzione particolare",
                    "content": r"\text{Determinata con il metodo dei coefficienti indeterminati o variazione dei parametri}"
                })
        else:
            steps.append({
                "title": "Metodo risolutivo",
                "content": r"\text{Equazione lineare a coefficienti costanti risolta con equazione caratteristica}"
            })
    except Exception:
        steps.append({
            "title": "Metodo risolutivo",
            "content": r"\text{Equazione lineare a coefficienti costanti risolta con equazione caratteristica}"
        })

    return steps


# ── Passaggi: Cauchy-Eulero ───────────────────────────────────────────
def _passaggi_eulero(eq, ordine):
    steps = []
    steps.append({
        "title": "Equazione di Cauchy-Eulero",
        "content": r"a\,x^2 y'' + b\,x\,y' + c\,y = 0"
    })
    steps.append({
        "title": "Sostituzione",
        "content": r"y = x^r \implies \text{equazione algebrica in } r"
    })
    r_sym = symbols('r')
    steps.append({
        "title": "Equazione indiciale",
        "content": r"a\,r(r-1) + b\,r + c = 0"
    })
    return steps


# ── Passaggi: Esatta ─────────────────────────────────────────────────
def _passaggi_esatta(lhs, rhs):
    steps = []
    steps.append({
        "title": "Verifica condizione di esattezza",
        "content": r"\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}"
    })
    steps.append({
        "title": "Ricerca della funzione potenziale",
        "content": r"F(x,y) \text{ tale che } \frac{\partial F}{\partial x} = M, \quad \frac{\partial F}{\partial y} = N"
    })
    steps.append({
        "title": "Soluzione implicita",
        "content": r"F(x,y) = C"
    })
    return steps


# ── Passaggi: Omogenea ────────────────────────────────────────────────
def _passaggi_omogenea(lhs, rhs):
    steps = []
    steps.append({
        "title": "Equazione omogenea: sostituzione",
        "content": r"v = \frac{y}{x} \implies y = vx, \quad y' = v + x\,v'"
    })
    steps.append({
        "title": "Equazione in variabili separabili in v",
        "content": r"x\,v' = f(v) - v \implies \frac{dv}{f(v)-v} = \frac{dx}{x}"
    })
    steps.append({
        "title": "Integrazione e ritorno a y",
        "content": r"\text{Si integra e si sostituisce } v = y/x"
    })
    return steps


# ══════════════════════════════════════════════════════════════════════════
#  FUNZIONE PRINCIPALE DI RISOLUZIONE
# ══════════════════════════════════════════════════════════════════════════
def risolvi_equazione_differenziale(input_utente, condizioni=None):
    """
    Funzione principale: riceve l'input dell'utente e restituisce il risultato.
    """
    try:
        # 1. Parsing dell'input
        eq, ordine, latex_input = parsifica_input(input_utente)

        # 2. Classificazione
        tipo_key, nome_tipo, hints = classifica_ode(eq)

        # 3. Risoluzione con SymPy dsolve
        try:
            soluzione = dsolve(eq, y)
        except Exception as e1:
            # Fallback: prova con hint specifici
            soluzione = None
            if hints:
                for hint in hints:
                    if hint.endswith('_Integral'):
                        continue
                    try:
                        soluzione = dsolve(eq, y, hint=hint)
                        break
                    except Exception:
                        continue

            if soluzione is None:
                # Ultimo tentativo: riscrittura
                try:
                    eq_alt = Eq(eq.lhs - eq.rhs, 0)
                    soluzione = dsolve(eq_alt, y)
                except Exception:
                    raise ValueError(
                        f"Impossibile risolvere questa equazione differenziale. "
                        f"Tipo rilevato: {nome_tipo}. Errore: {e1}"
                    )

        sol_generale = soluzione.rhs

        # 4. Soluzione particolare (Cauchy)
        sol_particolare = None
        if condizioni and 'x0' in condizioni and 'y0' in condizioni:
            try:
                ics = {y_func(condizioni['x0']): condizioni['y0']}
                if 'dy0' in condizioni:
                    ics[y_func(x).diff(x).subs(x, condizioni['x0'])] = condizioni['dy0']

                sol_cauchy = dsolve(eq, y, ics=ics)
                sol_particolare = sol_cauchy.rhs
            except Exception:
                # Fallback: risolvi manualmente per le costanti
                try:
                    sol_particolare = _risolvi_cauchy_manuale(
                        sol_generale, condizioni, ordine
                    )
                except Exception:
                    sol_particolare = None

        # 5. Genera passaggi
        passaggi = genera_passaggi(
            eq, tipo_key, ordine, sol_generale, sol_particolare, condizioni
        )

        # 6. Costruisci output
        latex_steps = []

        # Input ricevuto
        latex_steps.append({
            "title": "Input ricevuto",
            "content": latex_input
        })

        # Passaggi risolutivi
        latex_steps.extend(passaggi)

        return {"success": True, "latex": latex_steps}

    except Exception as e:
        print(f"[ERROR equazioni_differenziali] {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def _risolvi_cauchy_manuale(sol_gen, condizioni, ordine):
    """
    Fallback: applica le condizioni iniziali manualmente risolvendo
    per le costanti C1, C2, ...
    """
    x0_val = condizioni['x0']
    costanti = [s for s in sol_gen.free_symbols if str(s).startswith('C') and str(s)[1:].isdigit()]
    costanti = sorted(costanti, key=lambda s: int(str(s)[1:]))

    equazioni_cond = []

    # y(x0) = y0
    if 'y0' in condizioni:
        eq0 = Eq(sol_gen.subs(x, x0_val), condizioni['y0'])
        equazioni_cond.append(eq0)

    # y'(x0) = dy0
    if 'dy0' in condizioni:
        dsol = diff(sol_gen, x)
        eq1 = Eq(dsol.subs(x, x0_val), condizioni['dy0'])
        equazioni_cond.append(eq1)

    if not equazioni_cond:
        return None

    sol_cost = solve(equazioni_cond, costanti)
    if sol_cost:
        return simplify(sol_gen.subs(sol_cost))

    return None


# ══════════════════════════════════════════════════════════════════════════
#  WRAPPER RETROCOMPATIBILE (usato da condizioni_differenziali.py)
# ══════════════════════════════════════════════════════════════════════════
def solve_differential_equation(equation_str, conditions=None):
    """
    Wrapper di compatibilità con la vecchia API.
    Converte il formato P(Δ)y = u(t) nel nuovo formato y' + ... = ...
    e chiama risolvi_equazione_differenziale().
    """
    try:
        # Il vecchio formato è "P(Δ) * y = u(t)"
        # Proviamo prima a risolvere direttamente con il nuovo solver
        # Se fallisce, tentiamo una conversione dal formato operatore Δ
        import re as _re

        eq = equation_str.strip()

        # Converti notazione operatore Δ nel formato standard derivate
        # Es: "Δ² * y = ..." → "y'' = ..."
        eq = eq.replace('\\Delta', 'Δ')
        eq = eq.replace('Δ²', "y''").replace('Δ^2', "y''")
        eq = eq.replace('Δ³', "y'''").replace('Δ^3', "y'''")
        eq = _re.sub(r'Δ\^(\d+)', lambda m: "y" + "'" * int(m.group(1)), eq)
        eq = eq.replace('Δ', "y'")

        # Rimuovi * y se presente (era il formato P(Δ)*y)
        eq = eq.replace(" * y ", " ")
        eq = eq.replace("*y ", " ")
        eq = eq.replace(" *y", " ")

        # Sostituisci t con x per il nuovo solver
        eq = eq.replace('(t)', '(x)')
        # Non sostituire t dentro funzioni come exp(-2t) → exp(-2x)
        eq = _re.sub(r'\bt\b', 'x', eq)

        result = risolvi_equazione_differenziale(eq, conditions)
        return result

    except Exception as e:
        return {"success": False, "error": str(e)}
