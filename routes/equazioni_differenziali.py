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
import re, traceback, time
from collections import Counter

equazioni_differenziali_bp = Blueprint("equazioni_differenziali", __name__)
_transformations = standard_transformations + (implicit_multiplication_application, convert_xor,)

DEBUG = False  # Imposta a True per log dettagliati

def _log(msg):
    if DEBUG:
        print(f"[ODE DEBUG] {msg}")

# ═══════════════════ VARIABILE INDIPENDENTE ═══════════════════
def _detect_var(testo):
    clean = re.sub(r'(sin|cos|tan|sqrt|exp|text|latex|log|ln)\b', '', testo)
    if re.search(r'\bt\b', clean):
        return 't'
    return 'x'

def _get_symbols(var_name):
    v = symbols(var_name, real=True)
    yf = Function('y')
    return v, yf, yf(v)

# ═══════════════════ PREPROCESSING INPUT ═══════════════════
def _normalize(testo):
    s = testo.strip()
    s = s.replace('\\Delta', 'Δ').replace('Δ', 'd')
    s = s.replace('∘', ' ')
    s = s.replace('·', '*')
    s = s.replace('^', '**')
    
    # Unicode superscript esteso
    s = s.replace('⁰', '**0').replace('¹', '**1').replace('²', '**2')
    s = s.replace('³', '**3').replace('⁴', '**4').replace('⁵', '**5')
    s = s.replace('⁶', '**6').replace('⁷', '**7').replace('⁸', '**8').replace('⁹', '**9')
    
    # Aggiunge y mancante dopo un operatore (es. d²y senza parentesi o ∘)
    if 'y' not in s:
        if '=' in s:
            lhs, rhs = s.split('=', 1)
            lhs = lhs.strip()
            if lhs and lhs[-1].isdigit():
                lhs += '*y'
            elif lhs and lhs[-1] == ')':
                lhs += '*y'
            elif lhs and lhs[-1] == 'd':
                lhs += '*y'
            s = f"{lhs} = {rhs}"
        else:
            s += '*y'
            
    return s

def _is_operator_form(lhs):
    if "y'" in lhs or "y''" in lhs:
        return False
    if re.search(r'\bd\b', lhs) or re.search(r'd\s*\*\*', lhs) or re.search(r'd\s*\(', lhs):
        return True
    return False

def _prepare_op_str(s):
    s = s.strip()
    s = re.sub(r'[\s\*]*y\s*$', '', s).strip()
    s = s.rstrip('* ')
    if not s: return 'd'
    s = re.sub(r'\)\s*\(', ')*(', s)
    s = re.sub(r'\bd\s*\(', 'd*(', s)
    s = re.sub(r'\)\s*d\b', ')*d', s)
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)
    s = re.sub(r'\)\s*(\d)', r')*\1', s)
    return s

def _expand_operator(lhs_str, rhs_str, var_sym):
    op_str = _prepare_op_str(lhs_str)
    d_sym = symbols('d')
    local = {'d': d_sym, 'e': sp.E, 'pi': pi, 'E': sp.E, 'Rational': Rational}
    try:
        op_expr = parse_expr(op_str, local_dict=local, transformations=_transformations)
    except Exception as e:
        raise ValueError(f"Impossibile interpretare l'operatore '{op_str}': {e}")
    op_expanded = expand(op_expr)
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
    rhs_parsed = _parse_rhs(rhs_str, var_sym)
    eq = Eq(lhs_ode, rhs_parsed)
    return eq, degree, latex(eq)

def _parse_rhs(rhs_str, var_sym):
    s = rhs_str.strip()
    if not s or s == '0': return sp.Integer(0)
    var_name = str(var_sym)
    s = re.sub(r'(?<![a-zA-Z_\d])(\d+)\s*/\s*(\d+)(?![a-zA-Z_\d\(])', r'Rational(\1,\2)', s)
    s = re.sub(r'\be\s*\*\*\s*\(([^)]+)\)', r'exp(\1)', s)
    s = re.sub(r'\be\s*\*\*\s*([a-zA-Z0-9_]+)', r'exp(\1)', s)
    local = {var_name: var_sym, 'e': sp.E, 'E': sp.E, 'pi': pi, 'exp': exp,
             'sin': sin, 'cos': cos, 'tan': tan, 'ln': ln, 'log': log,
             'sqrt': sqrt, 'Rational': Rational, 'I': I}
    return parse_expr(s, local_dict=local, transformations=_transformations)

# ═══════════════════ SANITIZZAZIONE LATEX ═══════════════════
def sanitize_latex(s):
    s = s.replace('Δ', r'\Delta ')
    s = s.replace('∘', r'\circ ')
    s = s.replace('**', '^')
    s = s.replace('·', r'\cdot ')
    s = s.replace('*', r'\cdot ')
    
    s = s.replace('⁰', '^{0}').replace('¹', '^{1}').replace('²', '^{2}').replace('³', '^{3}')
    s = s.replace('⁴', '^{4}').replace('⁵', '^{5}').replace('⁶', '^{6}').replace('⁷', '^{7}')
    s = s.replace('⁸', '^{8}').replace('⁹', '^{9}')
    s = s.replace('₀', '_{0}').replace('₁', '_{1}').replace('₂', '_{2}').replace('₃', '_{3}')
    s = s.replace('₄', '_{4}').replace('₅', '_{5}').replace('₆', '_{6}').replace('⷇', '_{7}')
    s = s.replace('₈', '_{8}').replace('₉', '_{9}')
    
    s = re.sub(r'(?<![a-zA-Z\d])(\d+)\s*/\s*(\d+)', r'\\frac{\1}{\2}', s)
    
    # Funzioni matematiche
    s = re.sub(r'\barcsin\b', r'\\arcsin', s)
    s = re.sub(r'\barccos\b', r'\\arccos', s)
    s = re.sub(r'\barctan\b', r'\\arctan', s)
    s = re.sub(r'\bsin\b', r'\\sin', s)
    s = re.sub(r'\bcos\b', r'\\cos', s)
    s = re.sub(r'\btan\b', r'\\tan', s)
    s = re.sub(r'\blog\b', r'\\ln', s)
    s = re.sub(r'\bln\b', r'\\ln', s)
    
    s = re.sub(r'\bexp\s*\(([^)]+)\)', r'e^{\1}', s)
    s = re.sub(r'\be\s*\^\s*\(([^)]+)\)', r'e^{\1}', s)
    s = re.sub(r'\be\s*\^\s*([a-zA-Z0-9_\-]+)', r'e^{\1}', s)
    
    s = re.sub(r'\(\s*(\\frac{[^}]+}{[^}]+}[^)]*)\)', r'\\left(\1\\right)', s)
    return s.strip()

# ═══════════════════ PARSING FORMA STANDARD ═══════════════════
def _parse_standard(testo, var_sym):
    if '=' not in testo:
        raise ValueError("L'equazione deve contenere '='.")
    lhs_str, rhs_str = testo.split('=', 1)
    lhs_expr = _parse_ode_side(lhs_str.strip(), var_sym)
    rhs_expr = _parse_ode_side(rhs_str.strip(), var_sym)
    eq = Eq(lhs_expr, rhs_expr)
    ordine = 1
    if "y'''" in testo: ordine = 3
    elif "y''''" in testo: ordine = 4
    elif "y''" in testo: ordine = 2
    return eq, ordine, latex(eq)

def _parse_ode_side(s, var_sym):
    if not s.strip(): return sp.Integer(0)
    var_name = str(var_sym)
    yf = Function('y')
    y_var = yf(var_sym)
    s = re.sub(r'd\s*²\s*y\s*/\s*d\s*' + var_name + r'\s*²', "__DER2__", s)
    s = re.sub(r'd\s*2\s*y\s*/\s*d\s*' + var_name + r'\s*2', "__DER2__", s)
    s = re.sub(r'd\s*y\s*/\s*d\s*' + var_name, "__DER1__", s)
    s = s.replace("y''''", '__DER4__').replace("y'''", '__DER3__')
    s = s.replace("y''", '__DER2__').replace("y'", '__DER1__')
    s = re.sub(r'\by\b(?!_)', '__YFUNC__', s)
    s = re.sub(r'(?<![a-zA-Z_\d])(\d+)\s*/\s*(\d+)(?![a-zA-Z_\d\(])', r'Rational(\1,\2)', s)
    s = re.sub(r'(\d)(__DER|__YFUNC)', r'\1*\2', s)
    s = re.sub(r'(' + var_name + r')(__DER|__YFUNC)', r'\1*\2', s)
    s = re.sub(r'\)(__DER|__YFUNC)', r')*\1', s)
    s = re.sub(r'(__DER\d__|__YFUNC__)(' + var_name + r'|\()', r'\1*\2', s)
    s = re.sub(r'(__DER\d__|__YFUNC__)(\d)', r'\1*\2', s)
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

def parsifica_input(testo):
    _log(f"Input originale: {testo}")
    testo_orig = testo.strip()
    var_name = _detect_var(testo_orig)
    var_sym = symbols(var_name, real=True)
    testo_norm = _normalize(testo_orig)
    _log(f"Normalizzato: {testo_norm}")
    latex_sanitized = sanitize_latex(testo_orig)
    
    if '=' not in testo_norm:
        raise ValueError("L'equazione deve contenere '='.")
    lhs_raw, rhs_raw = testo_norm.split('=', 1)
    if _is_operator_form(lhs_raw):
        eq, ordine, latex_exp = _expand_operator(lhs_raw, rhs_raw, var_sym)
        _log(f"Equazione espansa: {eq}")
        return eq, ordine, var_sym, latex_sanitized, latex_exp
    else:
        eq, ordine, latex_eq = _parse_standard(testo_norm, var_sym)
        _log(f"Equazione standard: {eq}")
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

def _verifica_soluzione(eq, sol, var_sym, ordine):
    """Verifica che la soluzione soddisfi l'equazione."""
    y_sol = sol
    lhs_eval = eq.lhs.subs(Function('y')(var_sym), y_sol)
    for n in range(1, ordine + 1):
        lhs_eval = lhs_eval.subs(Derivative(Function('y')(var_sym), (var_sym, n)), 
                                  diff(y_sol, var_sym, n))
    rhs_eval = eq.rhs.subs(Function('y')(var_sym), y_sol)
    diff_simplified = simplify(lhs_eval - rhs_eval)
    return diff_simplified == 0, diff_simplified

def genera_passaggi(eq, tipo_key, ordine, var_sym, sol_gen, sol_part=None, cond=None):
    yf = Function('y')
    y_sym = yf(var_sym)
    v = str(var_sym)
    steps = []
    _, nome_tipo, _ = classifica_ode(eq, y_sym)
    steps.append(_step("Classificazione", rf"\text{{{nome_tipo} di ordine {ordine}}}"))
    
    if 'nth_linear_constant_coeff' in tipo_key or 'linear_constant' in tipo_key:
        steps += _steps_coeff_costanti(eq, ordine, var_sym, y_sym, sol_gen)
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
    
    steps.append(_step("Soluzione generale", rf"y({v}) = {latex(simplify(sol_gen))}"))
    
    # Verifica generale
    verificato, diff_val = _verifica_soluzione(eq, sol_gen, var_sym, ordine)
    if verificato:
        steps.append(_step("Verifica", rf"\text{{Sostituendo in equazione originale: }} L[y] = {latex(eq.rhs)} \quad \text{{✅ Verificato}}"))
    else:
        steps.append(_step("Verifica", rf"\text{{Controllo fallito analiticamente (potrebbero servire semplificazioni aggiuntive)}}"))
    
    if sol_part is not None and cond:
        parts = []
        if 'y0' in cond: parts.append(rf"y({latex(cond['x0'])}) = {latex(cond['y0'])}")
        if 'dy0' in cond: parts.append(rf"y'({latex(cond['x0'])}) = {latex(cond['dy0'])}")
        steps.append(_step("Condizioni iniziali", r", \quad ".join(parts)))
        steps.append(_step("Soluzione particolare (Cauchy)", rf"\boxed{{y({v}) = {latex(simplify(sol_part))}}}"))
    return steps

def _steps_coeff_costanti(eq, ordine, var_sym, y_sym, sol_gen):
    steps = []
    eq_expr = eq.lhs - eq.rhs
    v = str(var_sym)
    r_sym = symbols('r')
    coeffs = {}
    for n in range(ordine + 1):
        if n == 0: coeffs[0] = eq_expr.coeff(y_sym)
        else: coeffs[n] = eq_expr.coeff(Derivative(y_sym, (var_sym, n)))
        
    char_poly = sum(coeffs.get(n, 0) * r_sym**n for n in range(ordine + 1))
    char_poly = expand(char_poly)
    steps.append(_step("Equazione caratteristica", rf"{latex(char_poly)} = 0"))
    radici = solve(char_poly, r_sym)
    rad_str = ", ".join([rf"r = {latex(r)}" for r in radici])
    steps.append(_step("Radici", rad_str))
    
    try:
        sol_omogenea = dsolve(Eq(eq.lhs, 0), y_sym).rhs
    except:
        sol_omogenea = sp.Integer(0)
        
    if ordine >= 2:
        # Analizza le radici e la loro molteplicità
        radici_list = [simplify(r) for r in radici]
        conteggio = Counter(radici_list)
        
        # Costruisci la soluzione omogenea
        termini = []
        for r, molt in conteggio.items():
            if sp.im(r) == 0:
                # Radice reale
                if molt == 1:
                    termini.append(rf"C_{{{len(termini)+1}}} e^{{{latex(r)}{v}}}")
                else:
                    # Radice reale multipla
                    for k in range(molt):
                        idx = len(termini) + 1
                        if k == 0:
                            termini.append(rf"C_{{{idx}}} e^{{{latex(r)}{v}}}")
                        else:
                            termini.append(rf"C_{{{idx}}} {v}^{{{k}}} e^{{{latex(r)}{v}}}")
            else:
                # Radice complessa (e sua coniugata)
                if r == sp.conjugate(r) or sp.im(r) < 0:
                    continue  # Processa solo la radice con parte immaginaria positiva
                a = sp.re(r)
                b = sp.Abs(sp.im(r))
                if molt == 1:
                    if a == 0:
                        idx = len(termini) + 1
                        termini.append(rf"C_{{{idx}}} \cos({latex(b)}{v}) + C_{{{idx+1}}} \sin({latex(b)}{v})")
                    else:
                        idx = len(termini) + 1
                        termini.append(rf"e^{{{latex(a)}{v}}}\left(C_{{{idx}}} \cos({latex(b)}{v}) + C_{{{idx+1}}} \sin({latex(b)}{v})\right)")
                else:
                    # Radici complesse multiple (molto raro ma coperto)
                    for k in range(molt):
                        prefix = ""
                        if k > 0: prefix = rf"{v}^{{{k}}} "
                        idx = len(termini) + 1
                        if a == 0:
                            termini.append(rf"{prefix}\left(C_{{{idx}}} \cos({latex(b)}{v}) + C_{{{idx+1}}} \sin({latex(b)}{v})\right)")
                        else:
                            termini.append(rf"{prefix}e^{{{latex(a)}{v}}}\left(C_{{{idx}}} \cos({latex(b)}{v}) + C_{{{idx+1}}} \sin({latex(b)}{v})\right)")

        y_h_latex = " + ".join(termini).replace("+ -", "- ")
        if y_h_latex:
            steps.append(_step("Soluzione omogenea", rf"y_o({v}) = {y_h_latex}"))
        else:
            steps.append(_step("Soluzione omogenea", rf"y_o({v}) = {latex(sol_omogenea)}"))
    else:
        steps.append(_step("Soluzione omogenea", rf"y_o({v}) = {latex(sol_omogenea)}"))
    
    f_t = eq.rhs
    if f_t != 0 and not sp.simplify(f_t).is_zero:
        steps.append(_step("Termine forzante", rf"f({v}) = {latex(f_t)}"))
        
        # Rilevamento risonanza migliorato
        risonanza = False
        termine_risonante = None
        molteplicita_risonanza = 0
        
        # Estrai i termini additivi del forzante
        f_t_expanded = sp.expand(f_t)
        if isinstance(f_t_expanded, sp.Add):
            f_terms = f_t_expanded.args
        else:
            f_terms = [f_t_expanded]
        
        for term in f_terms:
            # Estrai la parte non-polinomiale (esponenziale, trigonometrica)
            term_no_coeff = term.as_coeff_Mul()[1]
            
            # Rimuovi eventuali fattori polinomiali t^n per il test di risonanza
            term_base = term_no_coeff
            t_power = 0
            if isinstance(term_no_coeff, sp.Mul):
                for factor in term_no_coeff.args:
                    if factor == var_sym or (isinstance(factor, sp.Pow) and factor.base == var_sym):
                        if factor == var_sym:
                            t_power = 1
                        else:
                            t_power = factor.exp
                        term_base = term_no_coeff / (var_sym**t_power)
                        break
            
            # Verifica se la base (senza t^n) è soluzione dell'omogenea
            L_term = sp.Integer(0)
            for n in range(ordine + 1):
                L_term += coeffs.get(n, 0) * sp.diff(term_base, var_sym, n)
            
            if sp.simplify(L_term) == 0:
                risonanza = True
                termine_risonante = term_base
                molteplicita_risonanza = t_power + 1  # molteplicità della radice
                break
        
        if risonanza:
            if molteplicita_risonanza > 1:
                steps.append(_step("Risonanza rilevata", 
                    rf"\text{{Il termine }} {latex(termine_risonante)} \text{{ è soluzione dell'omogenea (molteplicità }} {molteplicita_risonanza}\text{{).}}"))
            else:
                steps.append(_step("Risonanza rilevata", 
                    rf"\text{{Il termine }} {latex(termine_risonante)} \text{{ è soluzione dell'omogenea.}}"))
        
        y_p = sp.simplify(sol_gen - sol_omogenea)
        y_p_exp = sp.expand(y_p)
        
        if isinstance(y_p_exp, sp.Add):
            yp_terms = y_p_exp.args
        else:
            yp_terms = [y_p_exp]
            
        syms_str = 'A B C D E F G H K L M N'
        syms_abc = symbols(syms_str)
        
        forma_terms = []
        yp_assumed = sp.Integer(0)
        coeff_map = {}
        
        idx_sym = 0
        for term in yp_terms:
            # Separa coefficiente numerico dalla parte simbolica
            coeff_val, symbolic_part = term.as_coeff_Mul()
            if symbolic_part == 1:
                # Termine puramente numerico (costante)
                S = syms_abc[idx_sym % len(syms_abc)]
                forma_terms.append(rf"{S}")
                yp_assumed += S
                try:
                    coeff_map[S] = float(coeff_val) if coeff_val.is_number else coeff_val
                except:
                    coeff_map[S] = coeff_val
                idx_sym += 1
            else:
                # Termine con parte simbolica
                S = syms_abc[idx_sym % len(syms_abc)]
                forma_terms.append(rf"{S} \cdot {latex(symbolic_part)}")
                yp_assumed += S * symbolic_part
                try:
                    coeff_map[S] = float(coeff_val) if coeff_val.is_number else coeff_val
                except:
                    coeff_map[S] = coeff_val
                idx_sym += 1
            
        forma_latex = " + ".join(forma_terms).replace("+ -", "- ")
        steps.append(_step("Forma ipotizzata", rf"y_p({v}) = {forma_latex}"))
        
        # Sostituzione nell'ODE esplicita
        L_yp = sp.Integer(0)
        for n in range(ordine + 1):
            if n == 0:
                L_yp += coeffs.get(0, 0) * yp_assumed
            else:
                L_yp += coeffs.get(n, 0) * sp.diff(yp_assumed, var_sym, n)
                
        L_yp_exp = sp.expand(L_yp)
        steps.append(_step("Sostituzione nell'ODE", rf"L[y_p] = {latex(L_yp_exp)} = {latex(f_t)}"))
        
        # Coefficienti determinati
        if coeff_map:
            coeff_sol_str = ", \quad ".join([rf"{S} = {latex(val)}" for S, val in coeff_map.items()])
            steps.append(_step("Sistema risolvente", rf"\text{{Uguagliando i coefficienti si ottiene: }} {coeff_sol_str}"))
        
        steps.append(_step("Soluzione particolare", rf"y_p({v}) = {latex(y_p)}"))

    return steps

# ═══════════════════ RISOLUTORE PRINCIPALE ═══════════════════
def risolvi_equazione_differenziale(input_utente, condizioni=None):
    start_time = time.time()
    try:
        eq, ordine, var_sym, latex_input, latex_exp = parsifica_input(input_utente)
        yf = Function('y')
        y_sym = yf(var_sym)
        tipo_key, nome_tipo, hints = classifica_ode(eq, y_sym)
        _log(f"Tipo rilevato: {tipo_key}")
        
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
                    
        _log(f"Soluzione: {soluzione}")
        sol_gen = soluzione.rhs
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
                
        latex_steps = []
        latex_steps.append(_step("Input ricevuto", latex_input))
        if latex_exp:
            latex_steps.append(_step("Equazione espansa", latex_exp))
        latex_steps += genera_passaggi(eq, tipo_key, ordine, var_sym, sol_gen, sol_part, condizioni)
        
        calc_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "success": True, 
            "latex": latex_steps,
            "ordine": ordine,
            "tipo": tipo_key,
            "nome_tipo": nome_tipo,
            "variabile": str(var_sym),
            "soluzione_generale": latex(simplify(sol_gen)),
            "soluzione_particolare": latex(simplify(sol_part)) if sol_part else None,
            "tempo_calcolo": calc_time
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def _cauchy_manuale(sol_gen, cond, var_sym):
    x0 = cond['x0']
    consts = sorted(
        [s for s in sol_gen.free_symbols 
         if str(s).startswith('C') and len(str(s)) > 1 and str(s)[1:].isdigit()],
        key=lambda s: int(str(s)[1:])
    )
    if not consts:
        consts = [s for s in sol_gen.free_symbols 
                  if s != var_sym and not str(s).startswith('e') and not str(s).startswith('pi')]
    
    eqs = []
    if 'y0' in cond:
        eqs.append(Eq(sol_gen.subs(var_sym, x0), cond['y0']))
    if 'dy0' in cond:
        eqs.append(Eq(diff(sol_gen, var_sym).subs(var_sym, x0), cond['dy0']))
    
    if len(eqs) != len(consts):
        return None
    
    sol = solve(eqs, consts, dict=True)
    if sol:
        return simplify(sol_gen.subs(sol[0]))
    return None

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
