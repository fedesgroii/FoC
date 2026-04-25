"""
routes/equazioni_differenziali.py – Risolutore ODE completo.
Supporta notazione y'/y'', operatore d/Δ, variabili x e t.
"""
from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import (symbols, Function, Eq, dsolve, Derivative, exp, simplify,
    cos, sin, tan, ln, log, latex, expand, solve, I, pi, oo, classify_ode,
    sqrt, Rational, Abs, integrate, diff, Poly)
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor)
import re, traceback

equazioni_differenziali_bp = Blueprint("equazioni_differenziali", __name__)
_transformations = standard_transformations + (implicit_multiplication_application, convert_xor,)

# ═══════════════════ VARIABILE INDIPENDENTE ═══════════════════
def _detect_var(testo):
    """Rileva se la variabile indipendente è t o x."""
    clean = re.sub(r'(sin|cos|tan|sqrt|exp|text|latex)\b', '', testo)
    if re.search(r'\bt\b', clean):
        return 't'
    return 'x'

def _get_symbols(var_name):
    """Restituisce (var_sym, y_func_instance)."""
    v = symbols(var_name, real=True)
    yf = Function('y')
    return v, yf, yf(v)

# ═══════════════════ PREPROCESSING INPUT ═══════════════════
def _normalize(testo):
    """Normalizza notazione: Δ→d, ·→*, ^→**, ∘→''."""
    s = testo.strip()
    s = s.replace('\\Delta', 'Δ').replace('Δ', 'd')
    s = s.replace('∘', ' ')
    s = s.replace('·', '*')
    s = s.replace('^', '**')
    # Unicode superscript
    s = s.replace('²', '**2').replace('³', '**3').replace('⁴', '**4')
    return s

def _is_operator_form(lhs):
    """Controlla se LHS usa notazione operatore d."""
    if "y'" in lhs or "y''" in lhs:
        return False
    if re.search(r'\bd\b', lhs) or re.search(r'd\s*\*\*', lhs) or re.search(r'd\s*\(', lhs):
        return True
    return False

def _prepare_op_str(s):
    """Prepara stringa operatore per parsing SymPy: aggiunge * impliciti."""
    s = s.strip()
    # Rimuovi 'y' finale (e eventuale * o spazio prima)
    s = re.sub(r'[\s\*]*y\s*$', '', s).strip()
    # Rimuovi eventuale * finale rimasto
    s = s.rstrip('* ')
    if not s:
        return 'd'
    # )( → )*(
    s = re.sub(r'\)\s*\(', ')*(', s)
    # d( → d*( ma non d**
    s = re.sub(r'\bd\s*\(', 'd*(', s)
    # )d → )*d
    s = re.sub(r'\)\s*d\b', ')*d', s)
    # numero( → numero*(
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)
    # )numero → )*numero
    s = re.sub(r'\)\s*(\d)', r')*\1', s)
    # d**n( → d**n*(
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)
    return s

def _expand_operator(lhs_str, rhs_str, var_sym):
    """Espande forma operatore in equazione SymPy standard."""
    op_str = _prepare_op_str(lhs_str)
    d_sym = symbols('d')
    local = {'d': d_sym, 'e': sp.E, 'pi': pi, 'E': sp.E, 'Rational': Rational}
    try:
        op_expr = parse_expr(op_str, local_dict=local, transformations=_transformations)
    except Exception as e:
        raise ValueError(f"Impossibile interpretare l'operatore '{op_str}': {e}")
    op_expanded = expand(op_expr)
    # Estrai coefficienti del polinomio in d
    poly = Poly(op_expanded, d_sym)
    degree = poly.degree()
    yf = Function('y')
    y_var = yf(var_sym)
    lhs_ode = sp.Integer(0)
    for monom, coeff in poly.as_dict().items():
        n = monom[0]
        if n == 0:
            lhs_ode += coeff * y_var
        else:
            lhs_ode += coeff * Derivative(y_var, (var_sym, n))
    # Parse RHS
    rhs_parsed = _parse_rhs(rhs_str, var_sym)
    eq = Eq(lhs_ode, rhs_parsed)
    # LaTeX espansione
    latex_exp = latex(eq)
    return eq, degree, latex_exp

def _parse_rhs(rhs_str, var_sym):
    """Parsa il termine noto (RHS)."""
    s = rhs_str.strip()
    if not s or s == '0':
        return sp.Integer(0)
    var_name = str(var_sym)
    # Frazioni intere a/b → Rational(a,b)
    s = re.sub(r'(?<![a-zA-Z_\d])(\d+)\s*/\s*(\d+)(?![a-zA-Z_\d\(])', r'Rational(\1,\2)', s)
    # e**(...) → exp(...)
    s = re.sub(r'\be\s*\*\*\s*\(([^)]+)\)', r'exp(\1)', s)
    s = re.sub(r'\be\s*\*\*\s*([a-zA-Z0-9_]+)', r'exp(\1)', s)
    local = {var_name: var_sym, 'e': sp.E, 'E': sp.E, 'pi': pi, 'exp': exp,
             'sin': sin, 'cos': cos, 'tan': tan, 'ln': ln, 'log': log,
             'sqrt': sqrt, 'Rational': Rational, 'I': I}
    return parse_expr(s, local_dict=local, transformations=_transformations)

# ═══════════════════ PARSING FORMA STANDARD ═══════════════════
def _parse_standard(testo, var_sym):
    """Parsa equazioni in forma y', y'', dy/dx."""
    if '=' not in testo:
        raise ValueError("L'equazione deve contenere '='.")
    lhs_str, rhs_str = testo.split('=', 1)
    lhs_expr = _parse_ode_side(lhs_str.strip(), var_sym)
    rhs_expr = _parse_ode_side(rhs_str.strip(), var_sym)
    eq = Eq(lhs_expr, rhs_expr)
    # Ordine
    ordine = 1
    testo_check = testo
    if "y'''" in testo_check: ordine = 3
    elif "y''''" in testo_check: ordine = 4
    elif "y''" in testo_check: ordine = 2
    return eq, ordine, latex(eq)

def _parse_ode_side(s, var_sym):
    """Converte una parte (LHS o RHS) in espressione SymPy."""
    if not s.strip():
        return sp.Integer(0)
    var_name = str(var_sym)
    yf = Function('y')
    y_var = yf(var_sym)
    # dy/dx, d²y/dx²
    s = re.sub(r'd\s*²\s*y\s*/\s*d\s*' + var_name + r'\s*²', "__DER2__", s)
    s = re.sub(r'd\s*2\s*y\s*/\s*d\s*' + var_name + r'\s*2', "__DER2__", s)
    s = re.sub(r'd\s*y\s*/\s*d\s*' + var_name, "__DER1__", s)
    # y'''' y''' y'' y'
    s = s.replace("y''''", '__DER4__')
    s = s.replace("y'''", '__DER3__')
    s = s.replace("y''", '__DER2__')
    s = s.replace("y'", '__DER1__')
    s = re.sub(r'\by\b(?!_)', '__YFUNC__', s)
    # Frazioni intere
    s = re.sub(r'(?<![a-zA-Z_\d])(\d+)\s*/\s*(\d+)(?![a-zA-Z_\d\(])', r'Rational(\1,\2)', s)
    # Moltiplicazione implicita
    s = re.sub(r'(\d)(__DER|__YFUNC)', r'\1*\2', s)
    s = re.sub(r'(' + var_name + r')(__DER|__YFUNC)', r'\1*\2', s)
    s = re.sub(r'\)(__DER|__YFUNC)', r')*\1', s)
    s = re.sub(r'(__DER\d__|__YFUNC__)(' + var_name + r'|\()', r'\1*\2', s)
    s = re.sub(r'(__DER\d__|__YFUNC__)(\d)', r'\1*\2', s)
    # e**... → exp(...)
    s = re.sub(r'\be\s*\*\*\s*\(([^)]+)\)', r'exp(\1)', s)
    s = re.sub(r'\be\s*\*\*\s*([a-zA-Z0-9_\-]+)', r'exp(\1)', s)
    local = {var_name: var_sym, 'e': sp.E, 'E': sp.E, 'pi': pi, 'exp': exp,
             'sin': sin, 'cos': cos, 'tan': tan, 'ln': ln, 'log': log,
             'sqrt': sqrt, 'Rational': Rational, 'I': I,
             '__YFUNC__': y_var,
             '__DER1__': Derivative(y_var, var_sym),
             '__DER2__': Derivative(y_var, (var_sym, 2)),
             '__DER3__': Derivative(y_var, (var_sym, 3)),
             '__DER4__': Derivative(y_var, (var_sym, 4))}
    return parse_expr(s, local_dict=local, transformations=_transformations)

# ═══════════════════ SANITIZZAZIONE LATEX ═══════════════════
def sanitize_latex(s):
    """Sanitizza l'input utente per una visualizzazione perfetta in LaTeX."""
    s = s.replace('Δ', r'\Delta ')
    s = s.replace('∘', r'\circ ')
    s = s.replace('**', '^')
    s = s.replace('·', r'\cdot ')
    s = s.replace('*', r'\cdot ')
    
    # Apici e pedici unicode
    s = s.replace('⁰', '^{0}').replace('¹', '^{1}').replace('²', '^{2}').replace('³', '^{3}')
    s = s.replace('⁴', '^{4}').replace('⁵', '^{5}').replace('⁶', '^{6}').replace('⁷', '^{7}')
    s = s.replace('⁸', '^{8}').replace('⁹', '^{9}')
    
    s = s.replace('₀', '_{0}').replace('₁', '_{1}').replace('₂', '_{2}').replace('₃', '_{3}')
    s = s.replace('₄', '_{4}').replace('₅', '_{5}').replace('₆', '_{6}').replace('⷇', '_{7}')
    s = s.replace('₈', '_{8}').replace('₉', '_{9}')
    
    # Frazioni: a/b -> \frac{a}{b}
    s = re.sub(r'(?<![a-zA-Z\d])(\d+)\s*/\s*(\d+)', r'\\frac{\1}{\2}', s)
    
    # Funzioni matematiche
    s = re.sub(r'\bsin\b', r'\\sin', s)
    s = re.sub(r'\bcos\b', r'\\cos', s)
    s = re.sub(r'\btan\b', r'\\tan', s)
    s = re.sub(r'\bexp\s*\(([^)]+)\)', r'e^{\1}', s)
    s = re.sub(r'\be\s*\^\s*\(([^)]+)\)', r'e^{\1}', s)
    s = re.sub(r'\be\s*\^\s*([a-zA-Z0-9_\-]+)', r'e^{\1}', s)
    
    # Parentesi adattive attorno alle frazioni
    s = re.sub(r'\(\s*(\\frac{[^}]+}{[^}]+}[^)]*)\)', r'\\left(\1\\right)', s)
    
    return s.strip()


# ═══════════════════ PARSING PRINCIPALE ═══════════════════
def parsifica_input(testo):
    """Punto d'ingresso del parsing. Restituisce (eq, ordine, var_sym, latex_input, latex_espansione)."""
    testo_orig = testo.strip()
    var_name = _detect_var(testo_orig)
    var_sym = symbols(var_name, real=True)
    testo_norm = _normalize(testo_orig)
    
    latex_sanitized = sanitize_latex(testo_orig)
    
    if '=' not in testo_norm:
        raise ValueError("L'equazione deve contenere '='.")
    lhs_raw, rhs_raw = testo_norm.split('=', 1)
    if _is_operator_form(lhs_raw):
        eq, ordine, latex_exp = _expand_operator(lhs_raw, rhs_raw, var_sym)
        return eq, ordine, var_sym, latex_sanitized, latex_exp
    else:
        eq, ordine, latex_eq = _parse_standard(testo_norm, var_sym)
        return eq, ordine, var_sym, latex_sanitized, None

# ═══════════════════ CLASSIFICAZIONE ═══════════════════
_TIPO_NOMI = {
    'separable': 'Equazione a variabili separabili',
    '1st_linear': 'Equazione lineare del primo ordine',
    'Bernoulli': 'Equazione di Bernoulli',
    '1st_homogeneous_coeff': 'Equazione omogenea',
    '1st_exact': 'Equazione differenziale esatta',
    'nth_linear_constant_coeff_homogeneous': 'Eq. lineare a coeff. costanti omogenea',
    'nth_linear_constant_coeff_undetermined_coefficients': 'Eq. lineare a coeff. costanti non omogenea',
    'nth_linear_constant_coeff_variation_of_parameters': 'Eq. lineare a coeff. costanti (var. parametri)',
    'nth_linear_euler_eq_homogeneous': 'Equazione di Cauchy-Eulero omogenea',
    'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters': 'Eq. di Cauchy-Eulero non omogenea',
}

def classifica_ode(eq, y_sym):
    try:
        hints = classify_ode(eq, y_sym)
        if not hints:
            return 'generico', 'Equazione differenziale generica', []
        hints_list = list(hints)
        for h in hints_list:
            hc = h.replace('_Integral', '')
            for key, nome in _TIPO_NOMI.items():
                if key in hc:
                    return key, nome, hints_list
        return hints_list[0], 'Equazione differenziale', hints_list
    except Exception:
        return 'generico', 'Equazione differenziale generica', []

# ═══════════════════ PASSAGGI RISOLUTIVI ═══════════════════
def _step(title, content):
    return {"title": title, "content": content}

def genera_passaggi(eq, tipo_key, ordine, var_sym, sol_gen, sol_part=None, cond=None):
    yf = Function('y')
    y_sym = yf(var_sym)
    v = str(var_sym)
    steps = []
    _, nome_tipo, _ = classifica_ode(eq, y_sym)
    steps.append(_step("Classificazione", rf"\text{{{nome_tipo} di ordine {ordine}}}"))
    # Passaggi per tipo
    if 'nth_linear_constant_coeff' in tipo_key or 'linear_constant' in tipo_key:
        steps += _steps_coeff_costanti(eq, ordine, var_sym, y_sym)
    elif 'separable' in tipo_key:
        steps += [_step("Metodo", r"\text{Separazione delle variabili}"),
                  _step("Forma", r"\frac{dy}{g(y)} = f("+v+r")\,d"+v),
                  _step("Integrazione", r"\int \frac{dy}{g(y)} = \int f("+v+r")\,d"+v+" + C")]
    elif '1st_linear' in tipo_key:
        steps += [_step("Forma standard", r"y' + p("+v+r")y = q("+v+")"),
                  _step("Fattore integrante", r"\mu("+v+r") = e^{\int p("+v+r")\,d"+v+"}"),
                  _step("Soluzione", r"y = \frac{1}{\mu}\left[\int \mu\,q\,d"+v+r" + C\right]")]
    elif 'Bernoulli' in tipo_key:
        steps += [_step("Forma", r"y' + p("+v+r")y = q("+v+r")y^n"),
                  _step("Sostituzione", r"v = y^{1-n}"),
                  _step("Linearizzazione", r"v' + (1-n)p\,v = (1-n)q")]
    elif '1st_exact' in tipo_key:
        steps += [_step("Condizione", r"\frac{\partial M}{\partial y} = \frac{\partial N}{\partial "+v+"}"),
                  _step("Potenziale", r"F("+v+r",y) = C")]
    elif '1st_homogeneous' in tipo_key:
        steps += [_step("Sostituzione", r"v = y/"+v+r",\; y = v"+v),
                  _step("Separazione", r"\frac{dv}{f(v)-v} = \frac{d"+v+"}{"+v+"}")]
    elif 'euler' in tipo_key:
        steps += [_step("Tipo", r"\text{Equazione di Cauchy-Eulero}"),
                  _step("Sostituzione", r"y = "+v+r"^r \;\Rightarrow\; \text{eq. indiciale in } r")]
    else:
        steps.append(_step("Metodo", r"\text{Risoluzione simbolica con SymPy}"))
    # Soluzione generale
    steps.append(_step("Soluzione generale", rf"y({v}) = {latex(simplify(sol_gen))}"))
    # Cauchy
    if sol_part is not None and cond:
        parts = []
        if 'y0' in cond:
            parts.append(rf"y({latex(cond['x0'])}) = {latex(cond['y0'])}")
        if 'dy0' in cond:
            parts.append(rf"y'({latex(cond['x0'])}) = {latex(cond['dy0'])}")
        steps.append(_step("Condizioni iniziali", r", \quad ".join(parts)))
        steps.append(_step("Soluzione particolare (Cauchy)",
                           rf"\boxed{{y({v}) = {latex(simplify(sol_part))}}}"))
    return steps

def _steps_coeff_costanti(eq, ordine, var_sym, y_sym):
    """Passaggi per ODE lineari a coefficienti costanti."""
    steps = []
    eq_expr = eq.lhs - eq.rhs
    v = str(var_sym)
    r_sym = symbols('r')
    # Estrai coefficienti
    coeffs = {}
    for n in range(ordine + 1):
        if n == 0:
            coeffs[0] = eq_expr.coeff(y_sym)
        else:
            coeffs[n] = eq_expr.coeff(Derivative(y_sym, (var_sym, n)))
    # Polinomio caratteristico
    char_poly = sum(coeffs.get(n, 0) * r_sym**n for n in range(ordine + 1))
    char_poly = expand(char_poly)
    steps.append(_step("Equazione caratteristica", rf"{latex(char_poly)} = 0"))
    radici = solve(char_poly, r_sym)
    rad_str = ", ".join([rf"r = {latex(r)}" for r in radici])
    steps.append(_step("Radici", rad_str))
    # Tipo radici per ordine 2
    if ordine == 2 and len(radici) == 2:
        r1, r2 = radici
        if r1 != r2 and sp.im(r1) == 0 and sp.im(r2) == 0:
            steps.append(_step("Radici reali distinte",
                rf"y_o({v}) = C_1 e^{{{latex(r1)}{v}}} + C_2 e^{{{latex(r2)}{v}}}"))
        elif r1 == r2:
            steps.append(_step("Radice doppia",
                rf"y_o({v}) = (C_1 + C_2 {v})\,e^{{{latex(r1)}{v}}}"))
        elif sp.im(r1) != 0:
            a, b = sp.re(r1), sp.Abs(sp.im(r1))
            steps.append(_step("Radici complesse",
                rf"\alpha={latex(a)},\;\beta={latex(b)}"))
            if a == 0:
                steps.append(_step("Soluzione omogenea",
                    rf"y_o({v}) = C_1\cos({latex(b)}{v}) + C_2\sin({latex(b)}{v})"))
            else:
                steps.append(_step("Soluzione omogenea",
                    rf"y_o({v}) = e^{{{latex(a)}{v}}}\left(C_1\cos({latex(b)}{v}) + C_2\sin({latex(b)}{v})\right)"))
    # Termine forzante
    f_t = eq.rhs
    if f_t != 0 and not sp.simplify(f_t).is_zero:
        steps.append(_step("Termine forzante", rf"f({v}) = {latex(f_t)}"))
        steps.append(_step("Soluzione particolare",
            r"\text{Metodo dei coefficienti indeterminati / variazione dei parametri}"))
    return steps

# ═══════════════════ RISOLUTORE PRINCIPALE ═══════════════════
def risolvi_equazione_differenziale(input_utente, condizioni=None):
    try:
        eq, ordine, var_sym, latex_input, latex_exp = parsifica_input(input_utente)
        yf = Function('y')
        y_sym = yf(var_sym)
        tipo_key, nome_tipo, hints = classifica_ode(eq, y_sym)
        # Risoluzione
        soluzione = None
        try:
            soluzione = dsolve(eq, y_sym)
        except Exception:
            if hints:
                for h in hints:
                    if h.endswith('_Integral'): continue
                    try:
                        soluzione = dsolve(eq, y_sym, hint=h)
                        break
                    except Exception: continue
            if soluzione is None:
                try:
                    soluzione = dsolve(Eq(eq.lhs - eq.rhs, 0), y_sym)
                except Exception as e:
                    raise ValueError(f"Impossibile risolvere: {e}")
        sol_gen = soluzione.rhs
        # Cauchy
        sol_part = None
        if condizioni and 'x0' in condizioni and 'y0' in condizioni:
            try:
                ics = {yf(condizioni['x0']): condizioni['y0']}
                if 'dy0' in condizioni:
                    ics[yf(var_sym).diff(var_sym).subs(var_sym, condizioni['x0'])] = condizioni['dy0']
                sol_part = dsolve(eq, y_sym, ics=ics).rhs
            except Exception:
                try:
                    sol_part = _cauchy_manuale(sol_gen, condizioni, var_sym)
                except Exception: pass
        # Output
        latex_steps = []
        latex_steps.append(_step("Input ricevuto", latex_input))
        if latex_exp:
            latex_steps.append(_step("Equazione espansa", latex_exp))
        latex_steps += genera_passaggi(eq, tipo_key, ordine, var_sym, sol_gen, sol_part, condizioni)
        return {"success": True, "latex": latex_steps}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def _cauchy_manuale(sol_gen, cond, var_sym):
    x0 = cond['x0']
    consts = sorted([s for s in sol_gen.free_symbols if str(s).startswith('C') and str(s)[1:].isdigit()],
                    key=lambda s: int(str(s)[1:]))
    eqs = []
    if 'y0' in cond:
        eqs.append(Eq(sol_gen.subs(var_sym, x0), cond['y0']))
    if 'dy0' in cond:
        eqs.append(Eq(diff(sol_gen, var_sym).subs(var_sym, x0), cond['dy0']))
    sol = solve(eqs, consts)
    return simplify(sol_gen.subs(sol)) if sol else None

def _parse_valore(s):
    s = s.strip().replace('^', '**')
    return parse_expr(s, local_dict={'e': sp.E, 'pi': pi, 'Rational': Rational},
                      transformations=_transformations)

# ═══════════════════ ENDPOINT API ═══════════════════
@equazioni_differenziali_bp.route('/api/equazioni_differenziali', methods=['POST'])
def api_equazione_differenziale():
    data = request.get_json()
    equazione = data.get("equazione", "").strip()
    x0 = data.get("x0", "").strip()
    y0 = data.get("y0", "").strip()
    dy0 = data.get("dy0", "").strip()
    if not equazione:
        return jsonify({"success": False, "error": "Nessuna equazione fornita."})
    condizioni = {}
    try:
        if x0 and y0:
            condizioni['x0'] = _parse_valore(x0)
            condizioni['y0'] = _parse_valore(y0)
        if x0 and dy0:
            condizioni['x0'] = _parse_valore(x0)
            condizioni['dy0'] = _parse_valore(dy0)
    except Exception as e:
        return jsonify({"success": False, "error": f"Errore condizioni iniziali: {e}"})
    return jsonify(risolvi_equazione_differenziale(equazione, condizioni or None))

# ═══════════════════ WRAPPER RETROCOMPATIBILE ═══════════════════
def solve_differential_equation(equation_str, conditions=None):
    """Compatibilità con condizioni_differenziali.py."""
    try:
        eq = equation_str.strip()
        eq = eq.replace('\\Delta', 'Δ')
        eq = re.sub(r'\bt\b', 'x', eq.replace('(t)', '(x)'))
        return risolvi_equazione_differenziale(eq, conditions)
    except Exception as e:
        return {"success": False, "error": str(e)}
