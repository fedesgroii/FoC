"""
routes/equazioni_alle_differenze.py – Risolutore di equazioni alle differenze (recurrence relations).
Supporta shift y(n+1), operatore differenza Δ e operatore shift E.
"""
from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import (symbols, Function, Eq, rsolve, exp, simplify,
    cos, sin, tan, ln, log, latex, expand, solve, I, pi, oo,
    sqrt, Rational, Abs, diff, Poly)
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor)
import re, traceback, time
from collections import Counter

equazioni_alle_differenze_bp = Blueprint("equazioni_alle_differenze", __name__)
_transformations = standard_transformations + (implicit_multiplication_application, convert_xor,)

DEBUG = False

def _log(msg):
    if DEBUG:
        print(f"[DIFF DEBUG] {msg}")

# ═══════════════════ VARIABILE INDIPENDENTE ═══════════════════
def _detect_var(testo):
    clean = re.sub(r'(sin|cos|tan|sqrt|exp|text|latex|log|ln)\b', '', testo)
    if re.search(r'\bk\b', clean):
        return 'k'
    if re.search(r'\bt\b', clean):
        return 't'
    return 'n'

def _get_symbols(var_name):
    v = symbols(var_name, integer=True)
    yf = Function('y')
    return v, yf, yf(v)

# ═══════════════════ PREPROCESSING INPUT ═══════════════════
def _normalize(testo):
    s = testo.strip()
    s = s.replace('\\Delta', 'Δ')
    s = s.replace('∘', ' ')
    s = s.replace('·', '*')
    s = s.replace('^', '**')
    
    # Unicode superscript esteso
    s = s.replace('⁰', '**0').replace('¹', '**1').replace('²', '**2')
    s = s.replace('³', '**3').replace('⁴', '**4').replace('⁵', '**5')
    s = s.replace('⁶', '**6').replace('⁷', '**7').replace('⁸', '**8').replace('⁹', '**9')
    
    # Unicode subscript
    s = s.replace('₀', '_0').replace('₁', '_1').replace('₂', '_2').replace('₃', '_3')
    s = s.replace('₄', '_4').replace('₅', '_5').replace('₆', '_6').replace('⷇', '_7')
    s = s.replace('₈', '_8').replace('₉', '_9')
    
    # y_{n+1} o y_{n} o y_n -> y(n+1)
    s = re.sub(r'y_\{([^}]+)\}', r'y(\1)', s)
    s = re.sub(r'y_([a-zA-Z0-9]+)', r'y(\1)', s)
    s = re.sub(r'y\[([^\]]+)\]', r'y(\1)', s)
    
    # Aggiunge y_n mancante dopo un operatore (es. Δ² senza parentesi o ∘)
    if 'y' not in s:
        if '=' in s:
            lhs, rhs = s.split('=', 1)
            lhs = lhs.strip()
            if lhs and lhs[-1].isdigit():
                lhs += '*y'
            elif lhs and lhs[-1] == ')':
                lhs += '*y'
            elif lhs and lhs[-1] in ('Δ', 'E'):
                lhs += '*y'
            s = f"{lhs} = {rhs}"
        else:
            s += '*y'
            
    return s

def _is_operator_form(lhs):
    if "y(" in lhs:
        return False
    if re.search(r'\b[ΔE]\b', lhs) or re.search(r'[ΔE]\s*\*\*', lhs) or re.search(r'[ΔE]\s*\(', lhs):
        return True
    return False

def _prepare_op_str(s):
    s = s.strip()
    s = re.sub(r'[\s\*]*y\s*$', '', s).strip()
    s = s.rstrip('* ')
    if not s: return 'Δ'
    s = re.sub(r'\)\s*\(', ')*(', s)
    s = re.sub(r'\b([ΔE])\s*\(', r'\1*(', s)
    s = re.sub(r'\)\s*([ΔE])\b', r')*\1', s)
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)
    s = re.sub(r'\)\s*(\d)', r')*\1', s)
    return s

def _expand_operator(lhs_str, rhs_str, var_sym):
    op_str = _prepare_op_str(lhs_str)
    # Sostituisco Δ con (E - 1) prima di parsare
    op_str = op_str.replace('Δ', '(E-1)')
    
    E_sym = symbols('E')
    local = {'E': E_sym, 'e': sp.E, 'pi': pi, 'Rational': Rational}
    try:
        op_expr = parse_expr(op_str, local_dict=local, transformations=_transformations)
    except Exception as e:
        raise ValueError(f"Impossibile interpretare l'operatore '{op_str}': {e}")
        
    op_expanded = expand(op_expr)
    poly = Poly(op_expanded, E_sym)
    degree = poly.degree()
    yf = Function('y')
    
    lhs_eq = sp.Integer(0)
    for monom, coeff in poly.as_dict().items():
        n_shift = monom[0]
        lhs_eq += coeff * yf(var_sym + n_shift)
        
    rhs_parsed = _parse_rhs(rhs_str, var_sym)
    eq = Eq(lhs_eq, rhs_parsed)
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
    
    s = re.sub(r'(?<![a-zA-Z\d])(\d+)\s*/\s*(\d+)', r'\\frac{\1}{\2}', s)
    
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
    
    # Determino l'ordine
    yf = Function('y')
    shifts = []
    for term in eq.lhs.free_symbols | eq.rhs.free_symbols | {eq.lhs, eq.rhs}:
        pass # free_symbols non becca le function calls facilmente
    
    # Cerchiamo gli shift testualmente prima di un'analisi AST completa
    shifts_found = re.findall(r'y\(' + str(var_sym) + r'\s*\+\s*(\d+)\)', testo)
    
    max_shift = 0
    min_shift = 0
    
    shift_ints = [int(s) for s in shifts_found if s]
    if re.search(r'y\(' + str(var_sym) + r'\)', testo):
        shift_ints.append(0)
        
    if shift_ints:
        max_shift = max(shift_ints)
        min_shift = min(shift_ints)
    else:
        # Analisi fallback usando i termini
        eq_exp = expand(eq.lhs - eq.rhs)
        for term in eq_exp.args if isinstance(eq_exp, sp.Add) else [eq_exp]:
            for arg in term.args:
                if isinstance(arg, Function('y')):
                    arg_in = arg.args[0]
                    if arg_in == var_sym:
                        shifts.append(0)
                    else:
                        c = sp.simplify(arg_in - var_sym)
                        if c.is_number:
                            shifts.append(int(c))
        if shifts:
            max_shift = max(shifts)
            min_shift = min(shifts)
            
    ordine = max_shift - min_shift
    if ordine == 0:
        ordine = 1 # minimo fallback
        
    return eq, ordine, latex(eq)

def _parse_ode_side(s, var_sym):
    if not s.strip(): return sp.Integer(0)
    var_name = str(var_sym)
    yf = Function('y')
    y_var = yf(var_sym)
    
    # Protegge y(...)
    s = re.sub(r'\by\(([^)]+)\)', r'__YFUNC__(\1)', s)
    s = re.sub(r'(?<![a-zA-Z_\d])(\d+)\s*/\s*(\d+)(?![a-zA-Z_\d\(])', r'Rational(\1,\2)', s)
    s = re.sub(r'(\d)(__YFUNC__)', r'\1*\2', s)
    s = re.sub(r'(' + var_name + r')(__YFUNC__)', r'\1*\2', s)
    s = re.sub(r'\)(__YFUNC__)', r')*\1', s)
    
    local = {var_name: var_sym, 'e': sp.E, 'E': sp.E, 'pi': pi, 'exp': exp,
             'sin': sin, 'cos': cos, 'tan': tan, 'ln': ln, 'log': log,
             'sqrt': sqrt, 'Rational': Rational, 'I': I,
             '__YFUNC__': yf}
    return parse_expr(s, local_dict=local, transformations=_transformations)

def parsifica_input(testo):
    _log(f"Input originale: {testo}")
    testo_orig = testo.strip()
    var_name = _detect_var(testo_orig)
    var_sym = symbols(var_name, integer=True)
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
def classifica_eq_alle_differenze(eq, y_sym, ordine, var_sym):
    eq_exp = expand(eq.lhs - eq.rhs)
    
    # Controlla se a coefficienti costanti
    is_constant_coeff = True
    has_forcing_term = False
    
    for term in (eq_exp.args if isinstance(eq_exp, sp.Add) else [eq_exp]):
        has_y = term.has(Function('y'))
        if not has_y:
            has_forcing_term = True
        else:
            # Estrai coefficiente della y
            term_c = term
            for factor in (term.args if isinstance(term, sp.Mul) else [term]):
                if factor.has(Function('y')):
                    term_c = term / factor
                    break
            if term_c.has(var_sym):
                is_constant_coeff = False
                
    homog = "omogenea" if not has_forcing_term else "non omogenea"
    coeff = "a coefficienti costanti" if is_constant_coeff else "a coefficienti variabili"
    
    tipo_key = f"{ordine}th_linear"
    if is_constant_coeff:
        tipo_key += "_constant_coeff"
    if has_forcing_term:
        tipo_key += "_nonhomogeneous"
    else:
        tipo_key += "_homogeneous"
        
    nome_str = "del primo ordine" if ordine == 1 else "del secondo ordine" if ordine == 2 else f"di ordine {ordine}"
    nome = f"Equazione lineare {nome_str} {coeff} {homog}"
    
    return tipo_key, nome

# ═══════════════════ PASSAGGI RISOLUTIVI ═══════════════════
def _step(title, content):
    return {"title": title, "content": content}

def _verifica_soluzione(eq, sol, var_sym, ordine):
    """Verifica che la soluzione soddisfi l'equazione."""
    y_sol = sol
    lhs_eval = eq.lhs
    rhs_eval = eq.rhs
    
    yf = Function('y')
    # Sostituiamo ogni shift y(n+k) con la sol(n+k)
    # Cerchiamo tutti i termini y(n+k) in eq
    shifts_to_sub = {}
    for term in eq.lhs.free_symbols | eq.rhs.free_symbols | {eq.lhs, eq.rhs}:
        pass
        
    # Un metodo robusto: scansionare l'espressione con match
    def replacer(expr):
        if isinstance(expr, Function('y')):
            arg = expr.args[0]
            # sostituisci var_sym con arg in y_sol
            return y_sol.subs(var_sym, arg)
        elif expr.args:
            return expr.func(*[replacer(a) for a in expr.args])
        return expr
        
    lhs_eval = replacer(eq.lhs)
    rhs_eval = replacer(eq.rhs)
    
    diff_simplified = simplify(lhs_eval - rhs_eval)
    return diff_simplified == 0, diff_simplified

def genera_passaggi(eq, tipo_key, ordine, var_sym, sol_gen, sol_part=None, cond=None):
    yf = Function('y')
    y_sym = yf(var_sym)
    v = str(var_sym)
    steps = []
    
    tipo_key, nome_tipo = classifica_eq_alle_differenze(eq, y_sym, ordine, var_sym)
    steps.append(_step("Classificazione", rf"\text{{{nome_tipo}}}"))
    
    if 'constant_coeff' in tipo_key:
        steps += _steps_coeff_costanti(eq, ordine, var_sym, y_sym, sol_gen)
    else:
        steps.append(_step("Metodo", r"\text{Risoluzione simbolica con SymPy}"))
        
    # Sostituzione formale di C0, C1 con C_1, C_2
    sol_gen_clean = sol_gen
    const_idx = 1
    for sym in sol_gen.free_symbols:
        if str(sym).startswith('C') and str(sym)[1:].isdigit():
            sol_gen_clean = sol_gen_clean.subs(sym, symbols(f'C_{const_idx}'))
            const_idx += 1
            
    steps.append(_step("Soluzione generale", rf"y_{{{v}}} = {latex(simplify(sol_gen_clean))}"))
    
    # Verifica
    verificato, diff_val = _verifica_soluzione(eq, sol_gen, var_sym, ordine)
    if verificato:
        steps.append(_step("Verifica", rf"\text{{Sostituendo in equazione originale: }} L[y_{{{v}}}] = {latex(eq.rhs)} \quad \text{{✅ Verificato}}"))
    else:
        steps.append(_step("Verifica", rf"\text{{Controllo fallito analiticamente}}"))
        
    if sol_part is not None and cond:
        parts = []
        for i in range(len(cond)):
            if f'y{i}' in cond:
                parts.append(rf"y_{{{latex(i)}}} = {latex(cond[f'y{i}'])}")
        steps.append(_step("Condizioni iniziali", r", \quad ".join(parts)))
        steps.append(_step("Soluzione particolare (Cauchy)", rf"\boxed{{y_{{{v}}} = {latex(simplify(sol_part))}}}"))
        
    return steps

def _steps_coeff_costanti(eq, ordine, var_sym, y_sym, sol_gen):
    steps = []
    eq_expr = eq.lhs - eq.rhs
    v = str(var_sym)
    r_sym = symbols('r')
    
    # Estraiamo l'equazione caratteristica
    yf = Function('y')
    
    # Troviamo lo shift minimo per normalizzare l'equazione caratteristica
    eq_exp = expand(eq_expr)
    shifts = []
    def find_shifts(expr):
        if isinstance(expr, Function('y')):
            arg_in = expr.args[0]
            if arg_in == var_sym:
                shifts.append(0)
            else:
                c = sp.simplify(arg_in - var_sym)
                if c.is_number:
                    shifts.append(int(c))
        elif expr.args:
            for a in expr.args: find_shifts(a)
    find_shifts(eq_exp)
    min_shift = min(shifts) if shifts else 0
    
    coeffs = {}
    def extract_coeffs(expr):
        if isinstance(expr, sp.Add):
            for term in expr.args: extract_coeffs(term)
        else:
            c, r = expr.as_coeff_Mul()
            # r è y(n+k)
            for factor in (r.args if isinstance(r, sp.Mul) else [r]):
                if isinstance(factor, Function('y')):
                    arg_in = factor.args[0]
                    shift = sp.simplify(arg_in - var_sym)
                    k = int(shift) if shift.is_number else 0
                    coeffs[k - min_shift] = c * (expr / factor / c)
                    break
    extract_coeffs(eq_exp)
    
    char_poly = sum(coeffs.get(n, 0) * r_sym**n for n in range(ordine + 1))
    char_poly = expand(char_poly)
    steps.append(_step("Equazione caratteristica", rf"{latex(char_poly)} = 0"))
    radici = solve(char_poly, r_sym)
    rad_str = ", ".join([rf"r = {latex(r)}" for r in radici])
    steps.append(_step("Radici", rad_str))
    
    # Soluzione omogenea
    try:
        sol_omogenea = rsolve(Eq(eq.lhs, 0), y_sym)
        if sol_omogenea is None: sol_omogenea = sp.Integer(0)
    except:
        sol_omogenea = sp.Integer(0)
        
    radici_list = [simplify(r) for r in radici]
    conteggio = Counter(radici_list)
    
    termini = []
    c_idx = 1
    
    # Elabora radici
    for r, molt in conteggio.items():
        if sp.im(r) == 0:
            for k in range(molt):
                if k == 0:
                    termini.append(rf"C_{{{c_idx}}} \left({latex(r)}\right)^{{{v}}}")
                else:
                    termini.append(rf"C_{{{c_idx}}} {v}^{{{k}}} \left({latex(r)}\right)^{{{v}}}")
                c_idx += 1
        else:
            if r == sp.conjugate(r) or sp.im(r) < 0:
                continue
            rho = sp.Abs(r)
            theta = sp.arg(r)
            for k in range(molt):
                prefix = ""
                if k > 0: prefix = rf"{v}^{{{k}}} "
                termini.append(rf"{prefix} \left({latex(rho)}\right)^{{{v}}} \left(C_{{{c_idx}}} \cos({latex(theta)}{v}) + C_{{{c_idx+1}}} \sin({latex(theta)}{v})\right)")
                c_idx += 2
                
    y_h_latex = " + ".join(termini).replace("+ -", "- ")
    if y_h_latex:
        steps.append(_step("Soluzione omogenea", rf"y_h({v}) = {y_h_latex}"))
    else:
        # Pulisco sol_omogenea
        s_c = sol_omogenea
        idx = 1
        for s in s_c.free_symbols:
            if str(s).startswith('C'):
                s_c = s_c.subs(s, symbols(f'C_{idx}'))
                idx += 1
        steps.append(_step("Soluzione omogenea", rf"y_h({v}) = {latex(s_c)}"))
        
    f_t = eq.rhs
    if f_t != 0 and not sp.simplify(f_t).is_zero:
        steps.append(_step("Termine forzante", rf"f({v}) = {latex(f_t)}"))
        
        # Rilevamento risonanza
        risonanza = False
        termine_risonante = None
        molteplicita_risonanza = 0
        
        f_t_expanded = sp.expand(f_t)
        if isinstance(f_t_expanded, sp.Add):
            f_terms = f_t_expanded.args
        else:
            f_terms = [f_t_expanded]
            
        for term in f_terms:
            term_no_coeff = term.as_coeff_Mul()[1]
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
            
            # Applica L_term a term_base
            L_term = sp.Integer(0)
            def apply_L(expr):
                # operator su y
                res = sp.Integer(0)
                for k_s, c_s in coeffs.items():
                    res += c_s * expr.subs(var_sym, var_sym + k_s)
                return res
            L_term = apply_L(term_base)
            
            if sp.simplify(L_term) == 0:
                risonanza = True
                termine_risonante = term_base
                molteplicita_risonanza = t_power + 1
                break
                
        if risonanza:
            if molteplicita_risonanza > 1:
                steps.append(_step("Risonanza rilevata", rf"\text{{Il termine }} {latex(termine_risonante)} \text{{ è soluzione dell'omogenea (molteplicità }} {molteplicita_risonanza}\text{{).}}"))
            else:
                steps.append(_step("Risonanza rilevata", rf"\text{{Il termine }} {latex(termine_risonante)} \text{{ è soluzione dell'omogenea.}}"))
                
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
            coeff_val, symbolic_part = term.as_coeff_Mul()
            if symbolic_part == 1:
                S = syms_abc[idx_sym % len(syms_abc)]
                forma_terms.append(rf"{S}")
                yp_assumed += S
                try: coeff_map[S] = float(coeff_val) if coeff_val.is_number else coeff_val
                except: coeff_map[S] = coeff_val
                idx_sym += 1
            else:
                S = syms_abc[idx_sym % len(syms_abc)]
                forma_terms.append(rf"{S} \cdot {latex(symbolic_part)}")
                yp_assumed += S * symbolic_part
                try: coeff_map[S] = float(coeff_val) if coeff_val.is_number else coeff_val
                except: coeff_map[S] = coeff_val
                idx_sym += 1
                
        forma_latex = " + ".join(forma_terms).replace("+ -", "- ")
        steps.append(_step("Forma ipotizzata", rf"y_p({v}) = {forma_latex}"))
        
        L_yp = sp.Integer(0)
        for k_s, c_s in coeffs.items():
            L_yp += c_s * yp_assumed.subs(var_sym, var_sym + k_s)
            
        L_yp_exp = sp.expand(L_yp)
        steps.append(_step("Sostituzione nell'equazione", rf"L[y_p] = {latex(L_yp_exp)} = {latex(f_t)}"))
        
        if coeff_map:
            coeff_sol_str = ", \quad ".join([rf"{S} = {latex(val)}" for S, val in coeff_map.items()])
            steps.append(_step("Sistema risolvente", rf"\text{{Uguagliando i coefficienti si ottiene: }} {coeff_sol_str}"))
            
        steps.append(_step("Soluzione particolare", rf"y_p({v}) = {latex(y_p)}"))
        
    return steps

# ═══════════════════ RISOLUTORE PRINCIPALE ═══════════════════
def risolvi_eq_alle_differenze(input_utente, condizioni=None):
    start_time = time.time()
    try:
        eq, ordine, var_sym, latex_input, latex_exp = parsifica_input(input_utente)
        yf = Function('y')
        y_sym = yf(var_sym)
        tipo_key, nome_tipo = classifica_eq_alle_differenze(eq, y_sym, ordine, var_sym)
        _log(f"Tipo rilevato: {tipo_key}")
        
        soluzione = None
        ics = {}
        if condizioni:
            for i in range(10):
                if f'y{i}' in condizioni:
                    ics[yf(i)] = condizioni[f'y{i}']
                    
        try:
            if ics:
                soluzione = rsolve(eq, y_sym, ics)
                # se fallisce con ics, proviamo senza e sostituiamo manualmente
                if soluzione is None:
                    sol_gen_temp = rsolve(eq, y_sym)
                    if sol_gen_temp is not None:
                        soluzione = _cauchy_manuale(sol_gen_temp, condizioni, var_sym)
            else:
                soluzione = rsolve(eq, y_sym)
                
        except Exception as e:
            _log(f"rsolve failed: {e}")
            raise ValueError(f"Impossibile risolvere: {e}")
            
        if soluzione is None:
            raise ValueError("Impossibile trovare una soluzione analitica per questa equazione alle differenze.")
            
        _log(f"Soluzione: {soluzione}")
        
        # Determina sol_gen e sol_part
        if ics:
            sol_part = soluzione
            sol_gen = rsolve(eq, y_sym)
        else:
            sol_gen = soluzione
            sol_part = None
            
        latex_steps = []
        latex_steps.append(_step("Input ricevuto", latex_input))
        if latex_exp:
            latex_steps.append(_step("Equazione espansa", latex_exp))
        latex_steps += genera_passaggi(eq, tipo_key, ordine, var_sym, sol_gen, sol_part, condizioni)
        
        calc_time = round((time.time() - start_time) * 1000, 2)
        
        # Sostituzione formale di C0, C1
        sol_gen_clean = sol_gen
        const_idx = 1
        for sym in sol_gen.free_symbols:
            if str(sym).startswith('C') and str(sym)[1:].isdigit():
                sol_gen_clean = sol_gen_clean.subs(sym, symbols(f'C_{const_idx}'))
                const_idx += 1
                
        return {
            "success": True, 
            "latex": latex_steps,
            "ordine": ordine,
            "tipo": tipo_key,
            "nome_tipo": nome_tipo,
            "variabile": str(var_sym),
            "soluzione_generale": latex(simplify(sol_gen_clean)),
            "soluzione_particolare": latex(simplify(sol_part)) if sol_part else None,
            "tempo_calcolo": calc_time
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def _cauchy_manuale(sol_gen, cond, var_sym):
    consts = sorted(
        [s for s in sol_gen.free_symbols 
         if str(s).startswith('C') and str(s)[1:].isdigit()],
        key=lambda s: int(str(s)[1:])
    )
    
    eqs = []
    for i in range(10):
        if f'y{i}' in cond:
            eqs.append(Eq(sol_gen.subs(var_sym, i), cond[f'y{i}']))
            
    if len(eqs) != len(consts):
        return sol_gen # Non completamente determinata
        
    sol = solve(eqs, consts, dict=True)
    if sol:
        return simplify(sol_gen.subs(sol[0]))
    return None

def _parse_valore(s):
    s = s.strip().replace('^', '**')
    return parse_expr(s, local_dict={'e': sp.E, 'pi': pi, 'Rational': Rational},
                      transformations=_transformations)

# ═══════════════════ ENDPOINT API ═══════════════════
@equazioni_alle_differenze_bp.route('/api/equazioni_alle_differenze', methods=['POST'])
def api_eq_alle_differenze():
    data = request.get_json()
    equazione = data.get("equazione", "").strip()
    if not equazione:
        return jsonify({"success": False, "error": "Nessuna equazione fornita."})
        
    condizioni = {}
    try:
        # Può ricevere y0, y1, y2...
        for i in range(10):
            val = data.get(f"y{i}", "").strip()
            if val:
                condizioni[f"y{i}"] = _parse_valore(val)
    except Exception as e:
        return jsonify({"success": False, "error": f"Errore condizioni iniziali: {e}"})
        
    return jsonify(risolvi_eq_alle_differenze(equazione, condizioni or None))

# ═══════════════════ WRAPPER RETROCOMPATIBILE ═══════════════════
def solve_difference_equation(equation_str, conditions=None):
    try:
        return risolvi_eq_alle_differenze(equation_str, conditions)
    except Exception as e:
        return {"success": False, "error": str(e)}
