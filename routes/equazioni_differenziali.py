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
def _clean_ode_sol(expr, var_sym):
    """
    Pulisce la soluzione evitando che SymPy combini A*sin(wt) + B*cos(wt) 
    in una singola funzione sfasata.
    """
    if expr is None: return None
    
    # 1. Espande per cancellare termini (es. C1 - C1)
    expr = sp.expand(expr)
    
    def expand_trig_sums(e):
        if isinstance(e, (sp.sin, sp.cos)) and isinstance(e.args[0], sp.Add):
            arg = e.args[0]
            # Separiamo parte variabile e parte costante dell'argomento
            v_part = sp.Add(*[t for t in arg.args if t.has(var_sym)])
            c_part = sp.Add(*[t for t in arg.args if not t.has(var_sym)])
            if c_part != 0:
                if isinstance(e, sp.sin):
                    # sin(v + c) = sin(v)cos(c) + cos(v)sin(c)
                    return sp.sin(v_part)*sp.cos(c_part) + sp.cos(v_part)*sp.sin(c_part)
                else:
                    # cos(v + c) = cos(v)cos(c) - sin(v)sin(c)
                    return sp.cos(v_part)*sp.cos(c_part) - sp.sin(v_part)*sp.sin(c_part)
        return e

    # 2. Applica espansione manuale per somme trig sin(wt + phi)
    expr = expr.replace(lambda x: isinstance(x, (sp.sin, sp.cos)), expand_trig_sums)
    
    # 3. Espande di nuovo per distribuire coefficienti e semplificare costanti numeriche
    return sp.expand(expr)

def _detect_var(testo):
    clean = re.sub(r'(sin|cos|tan|sqrt|exp|text|latex|log|ln)\b', '', testo)
    if re.search(r'\bt\b', clean):
        return 't'
    return 'x'

def _normalize_variables(testo, var_target):
    """
    Converte tutte le variabili indipendenti (x <-> t) nella variabile target rilevata.
    """
    if var_target == 'x':
        return re.sub(r'\bt\b', 'x', testo)
    else:
        return re.sub(r'\bx\b', 't', testo)

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
    
    # Rimuove spazi intorno all'esponente per evitare errori di parsing in d^(n)
    s = re.sub(r'\s*\^\s*\(', '^(', s)
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
    
    # Pulizia finale esponenti per SymPy
    s = s.replace('** ', '**').replace(' **', '**')
    s = re.sub(r'\*\*\s*\(\s*([^)]+?)\s*\)', r'**(\1)', s)
    
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
    if "y''''" in testo: ordine = 4
    elif "y'''" in testo: ordine = 3
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
    
    # Normalizza le variabili (x <-> t) in base a quella rilevata
    testo_orig = _normalize_variables(testo_orig, var_name)
    
    testo_norm = _normalize(testo_orig)
    _log(f"Normalizzato: {testo_norm}")
    latex_sanitized = sanitize_latex(testo_orig)
    
    if '=' not in testo_norm:
        raise ValueError("L'equazione deve contenere '='.")
    lhs_raw, rhs_raw = testo_norm.split('=', 1)
    if _is_operator_form(lhs_raw):
        eq, ordine, latex_exp = _expand_operator(lhs_raw, rhs_raw, var_sym)
        _log(f"Equazione espansa: {eq}")
        return eq, ordine, var_sym, latex(eq), latex_exp
    else:
        eq, ordine, latex_eq = _parse_standard(testo_norm, var_sym)
        _log(f"Equazione standard: {eq}")
        return eq, ordine, var_sym, latex(eq), None

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

def _get_constants(expr):
    """Estrae le costanti C1, C2, C_1, etc. in ordine numerico."""
    consts = []
    for s in expr.free_symbols:
        name = str(s)
        match = re.match(r'^C_?(\d+)$', name)
        if match:
            consts.append((s, int(match.group(1))))
    return [c[0] for c in sorted(consts, key=lambda x: x[1])]

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

def _analyze_forcing_term(term, var_sym):
    term = expand(term)
    if not term.has(var_sym):
        return sp.Integer(0), 0
        
    a, b = term.as_coeff_Mul()
    factors = sp.Mul.make_args(b)
    
    p = sp.Integer(0)
    omega = sp.Integer(0)
    poly_degree = 0
    
    for f in factors:
        if f == var_sym:
            poly_degree = max(poly_degree, 1)
        elif isinstance(f, sp.Pow) and f.base == var_sym and f.exp.is_Integer and f.exp > 0:
            poly_degree = max(poly_degree, int(f.exp))
        elif isinstance(f, sp.exp):
            arg = f.args[0]
            p = diff(arg, var_sym)
        elif isinstance(f, sp.sin) or isinstance(f, sp.cos):
            arg = f.args[0]
            omega = diff(arg, var_sym)
            
    lambda_val = simplify(p + I * Abs(omega))
    return lambda_val, poly_degree

def _solve_for_constants_generic(sol_general, order, var_sym):
    consts = _get_constants(sol_general)
    if not consts or len(consts) != order:
        return {}, []
    
    steps = []
    y_syms = [symbols(f'y_{i}') for i in range(order)]
    
    eqs = []
    for i in range(order):
        if i == 0:
            der = sol_general
        else:
            der = diff(sol_general, var_sym, i)
        
        val_0 = der.subs(var_sym, 0)
        eqs.append(Eq(val_0, y_syms[i]))
    
    syst_latex = []
    for i, eq in enumerate(eqs):
        syst_latex.append(f"{latex(eq.lhs)} &= {latex(eq.rhs)}")
        
    steps.append(_step("Condizioni iniziali generiche in $t=0$",
        r" \begin{cases} " + r" \\ ".join(syst_latex) + r" \end{cases} "))
        
    sol = solve(eqs, consts, dict=True)
    if not sol:
        return {}, steps
        
    sol_dict = sol[0]
    sol_latex = []
    for c in consts:
        if c in sol_dict:
            sol_latex.append(f"{latex(c)} &= {latex(simplify(sol_dict[c]))}")
            
    steps.append(_step("Soluzione del sistema per le costanti",
        r" \begin{cases} " + r" \\ ".join(sol_latex) + r" \end{cases} "))
        
    return sol_dict, steps

def genera_passaggi(eq, tipo_key, ordine, var_sym, sol_gen, sol_part=None, cond=None):
    yf = Function('y')
    y_sym = yf(var_sym)
    v = str(var_sym)
    steps = []
    _, nome_tipo, _ = classifica_ode(eq, y_sym)
    steps.append(_step("Classificazione", rf"\text{{{nome_tipo} di ordine {ordine}}}"))
    
    sol_gen_iniziale = sol_gen
    if 'nth_linear_constant_coeff' in tipo_key or 'linear_constant' in tipo_key:
        new_steps, sol_gen = _steps_coeff_costanti(eq, ordine, var_sym, y_sym, sol_gen)
        steps += new_steps
        # Se la soluzione generale è stata aggiornata, ricalcoliamo la particolare (Cauchy) se necessario
        if sol_gen != sol_gen_iniziale and cond:
            try:
                sol_part = _cauchy_manuale(sol_gen, cond, var_sym)
            except: pass
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
    
    steps.append(_step("Soluzione generale", rf"y({v}) = {latex(_clean_ode_sol(sol_gen, var_sym))}"))
    
    # Verifica generale
    verificato, diff_val = _verifica_soluzione(eq, sol_gen, var_sym, ordine)
    if verificato:
        steps.append(_step("Verifica", rf"\text{{Sostituendo in equazione originale: }} L[y] = {latex(eq.rhs)} \quad \text{{✅ Verificato}}"))
    else:
        steps.append(_step("Verifica", rf"\text{{Controllo fallito analiticamente (potrebbero servire semplificazioni aggiuntive)}}"))
    
    if cond and sol_part is not None:
        parts = []
        if 'y0' in cond: parts.append(rf"y({latex(cond['x0'])}) = {latex(cond['y0'])}")
        if 'dy0' in cond: parts.append(rf"y'({latex(cond['x0'])}) = {latex(cond['dy0'])}")
        steps.append(_step("Condizioni iniziali", r", \quad ".join(parts)))
        steps.append(_step("Soluzione particolare (Cauchy)", rf"\boxed{{y({v}) = {latex(_clean_ode_sol(sol_part, var_sym))}}}"))
    elif not cond:
        sol_dict, generic_steps = _solve_for_constants_generic(sol_gen, ordine, var_sym)
        if sol_dict:
            sol_gen_sub = sol_gen.subs(sol_dict)
            sol_gen_sub = _clean_ode_sol(sol_gen_sub, var_sym)
            
            y_syms = [symbols(f'y_{i}') for i in range(ordine)]
            y_libera = sp.Integer(0)
            
            for y_i in y_syms:
                coeff = sp.diff(sol_gen_sub, y_i)
                y_libera += coeff * y_i
                
            y_forzata = _clean_ode_sol(sol_gen_sub - y_libera, var_sym)
            
            steps.extend(generic_steps)
            steps.append(_step("Soluzione con costanti esplicitate", 
                rf"y({var_sym}) = {latex(sol_gen_sub)}"))
                
            if y_forzata != 0 and y_libera != 0:
                steps.append(_step("Separazione", 
                    rf"\begin{{align}} y_{{\text{{libera}}}}({var_sym}) &= {latex(y_libera)} \\ y_{{\text{{forzata}}}}({var_sym}) &= {latex(y_forzata)} \end{{align}}"))
            elif y_libera != 0:
                steps.append(_step("Componenti della soluzione", 
                    rf"y({var_sym}) = y_{{\text{{libera}}}}({var_sym}) = {latex(y_libera)}"))
            
            # Aggiungi legenda
            legenda_items = []
            for i in range(ordine):
                if i == 0: der_str = "y(0)"
                elif i == 1: der_str = "y'(0)"
                elif i == 2: der_str = "y''(0)"
                else: der_str = rf"y^{{({i})}}(0)"
                legenda_items.append(rf"y_{{{i}}} = {der_str}")
            
            steps.append(_step("Legenda", rf"\text{{Dove: }} {', '.join(legenda_items)}"))

    return steps, sol_gen, sol_part

def _steps_coeff_costanti(eq, ordine, var_sym, y_sym, sol_gen):
    steps = []
    eq_expr = eq.lhs - eq.rhs
    v = str(var_sym)
    sol_omogenea = sp.Integer(0) # Inizializzazione per sicurezza
    r_sym = symbols('r')
    coeffs = {}
    for n in range(ordine + 1):
        if n == 0: coeffs[0] = eq_expr.coeff(y_sym)
        else: coeffs[n] = eq_expr.coeff(Derivative(y_sym, (var_sym, n)))
        
    char_poly = sum(coeffs.get(n, 0) * r_sym**n for n in range(ordine + 1))
    char_poly = expand(char_poly)
    steps.append(_step("Equazione caratteristica", rf"{latex(char_poly)} = 0"))
    radici_dict = sp.roots(char_poly, r_sym)
    radici_list = []
    rad_strs = []
    for r, m in radici_dict.items():
        if m > 1:
            rad_strs.append(rf"r = {latex(r)} \text{{ (molt. }} {m} \text{{)}}")
            for _ in range(m):
                radici_list.append(r)
        else:
            rad_strs.append(rf"r = {latex(r)}")
            radici_list.append(r)
            
    rad_str = ", ".join(rad_strs)
    steps.append(_step("Radici", rad_str))
    
    try:
        sol_omogenea = dsolve(Eq(eq.lhs, 0), y_sym).rhs
    except:
        sol_omogenea = sp.Integer(0)
        
    if ordine >= 2:
        # Analizza le radici e la loro molteplicità
        radici_list = [simplify(r) for r in radici_list]
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
        
        f_t_expanded = sp.expand(f_t)
        if isinstance(f_t_expanded, sp.Add):
            f_terms = f_t_expanded.args
        else:
            f_terms = [f_t_expanded]
            
        term_groups = {}
        for term in f_terms:
            lambda_val, deg = _analyze_forcing_term(term, var_sym)
            key = simplify(lambda_val)
            if sp.im(key) < 0:
                key = sp.conjugate(key)
            if key not in term_groups:
                term_groups[key] = []
            term_groups[key].append(term)
            
        syms_str = 'A B C D E F G H K L M N P Q R S T U V W Z'
        syms_abc = symbols(syms_str)
        idx_sym = 0
        
        yp_assumed_total = sp.Integer(0)
        
        for lambda_val, terms in term_groups.items():
            max_deg = max([_analyze_forcing_term(t, var_sym)[1] for t in terms])
            
            k = 0
            for r, m in radici_dict.items():
                if simplify(r - lambda_val) == 0 or simplify(r - sp.conjugate(lambda_val)) == 0:
                    k = max(k, m)
            
            resonance_factor = var_sym**k if k > 0 else sp.Integer(1)
            
            if k > 0:
                steps.append(_step("Risonanza rilevata", 
                    rf"\text{{Per termini con }} \lambda = {latex(lambda_val)} \text{{ c'è risonanza (molteplicità }} {k} \text{{). Si moltiplica per }} {v}^{{{k}}}\text{{.}}"))
                    
            p = sp.re(lambda_val)
            omega = sp.Abs(sp.im(lambda_val))
            
            if omega != 0:
                Q_n = sp.Integer(0)
                R_n = sp.Integer(0)
                for i in range(max_deg, -1, -1):
                    Sq = syms_abc[idx_sym % len(syms_abc)]
                    idx_sym += 1
                    Q_n += Sq * (var_sym**i)
                    
                    Sr = syms_abc[idx_sym % len(syms_abc)]
                    idx_sym += 1
                    R_n += Sr * (var_sym**i)
                yp_assumed = (Q_n * sp.cos(omega*var_sym) + R_n * sp.sin(omega*var_sym))
                if p != 0:
                    yp_assumed = yp_assumed * sp.exp(p*var_sym)
            else:
                P_n = sp.Integer(0)
                for i in range(max_deg, -1, -1):
                    S = syms_abc[idx_sym % len(syms_abc)]
                    idx_sym += 1
                    P_n += S * (var_sym**i)
                yp_assumed = P_n
                if p != 0:
                    yp_assumed = yp_assumed * sp.exp(p*var_sym)
                    
            yp_assumed = yp_assumed * resonance_factor
            yp_assumed_total += yp_assumed
            
        steps.append(_step("Forma ipotizzata (con metodo di somiglianza)", rf"y_p({v}) = {latex(yp_assumed_total)}"))
        
        L_yp = sp.Integer(0)
        for n in range(ordine + 1):
            if n == 0:
                L_yp += coeffs.get(0, 0) * yp_assumed_total
            else:
                L_yp += coeffs.get(n, 0) * sp.diff(yp_assumed_total, var_sym, n)
                
        L_yp_exp = sp.expand(L_yp)
        steps.append(_step("Sostituzione nell'ODE", rf"L[y_p] = {latex(L_yp_exp)} = {latex(f_t)}"))
        
        # ═══════════════════ RISOLVI IL SISTEMA PER I COEFFICIENTI ═══════════════════
        # Identifica i coefficienti incogniti (A, B, C, etc.)
        unknowns = [s for s in yp_assumed_total.free_symbols if str(s) in 'ABCDEFGHKLMNPQRSTUWZ']
        
        risolto_manualmente = False
        if unknowns:
            try:
                diff_expr = sp.expand(L_yp_exp - f_t)
                eqs_system = []
                
                # Raccogliamo i termini rispetto alle basi trovate in term_groups
                for lambda_val, terms in term_groups.items():
                    p = sp.re(lambda_val)
                    omega = sp.Abs(sp.im(lambda_val))
                    max_deg_val = max([_analyze_forcing_term(t, var_sym)[1] for t in terms])
                    
                    # Generiamo le basi per questo gruppo
                    group_bases = []
                    if omega != 0:
                        for i in range(max_deg_val + 1):
                            base_cos = (var_sym**i) * sp.cos(omega*var_sym)
                            base_sin = (var_sym**i) * sp.sin(omega*var_sym)
                            if p != 0:
                                base_cos *= sp.exp(p*var_sym)
                                base_sin *= sp.exp(p*var_sym)
                            group_bases.extend([base_cos, base_sin])
                    else:
                        for i in range(max_deg_val + 1):
                            base_poly = (var_sym**i)
                            if p != 0:
                                base_poly *= sp.exp(p*var_sym)
                            group_bases.append(base_poly)
                    
                    for b in group_bases:
                        coeff = diff_expr.coeff(b)
                        if coeff != 0 and not coeff.has(var_sym):
                            eqs_system.append(Eq(coeff, 0))
                
                # Aggiungiamo il termine costante se presente
                const_part = diff_expr.as_independent(var_sym, [sp.sin, sp.cos, sp.exp])[0]
                if const_part != 0 and any(u in const_part.free_symbols for u in unknowns):
                    eqs_system.append(Eq(const_part, 0))

                # Rimuovi duplicati e triviali
                eqs_system = list(set(eqs_system))
                eqs_system = [e for e in eqs_system if e != True and e.lhs != 0]

                if eqs_system:
                    sol_coeffs = solve(eqs_system, unknowns, dict=True)
                    if sol_coeffs:
                        y_p_calcolata = yp_assumed_total.subs(sol_coeffs[0])
                        y_p_calcolata = _clean_ode_sol(y_p_calcolata, var_sym)
                        
                        steps.append(_step("Sistema per i coefficienti", 
                            rf"\text{{Uguagliando i coefficienti delle funzioni base: }} {latex(eqs_system)}"))
                        
                        sol_str = ", ".join([rf"{latex(k)} = {latex(v)}" for k, v in sol_coeffs[0].items()])
                        steps.append(_step("Coefficienti determinati", rf"{sol_str}"))
                        
                        steps.append(_step("Soluzione particolare determinata", 
                            rf"y_p({v}) = {latex(y_p_calcolata)}"))
                        
                        # Aggiorna sol_gen per i passaggi successivi
                        sol_gen = sol_omogenea + y_p_calcolata
                        risolto_manualmente = True
            except Exception as e:
                _log(f"Errore risoluzione manuale yp: {e}")

        if not risolto_manualmente:
            y_p = _clean_ode_sol(sol_gen - sol_omogenea, var_sym)
            steps.append(_step("Soluzione particolare determinata", rf"y_p({v}) = {latex(y_p)}"))

    return steps, sol_gen

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
                x0_val = condizioni['x0']
                ics = {yf(x0_val): condizioni['y0']}
                if 'dy0' in condizioni:
                    ics[yf(var_sym).diff(var_sym).subs(var_sym, x0_val)] = condizioni['dy0']
                if 'd2y0' in condizioni:
                    ics[yf(var_sym).diff(var_sym, 2).subs(var_sym, x0_val)] = condizioni['d2y0']
                if 'd3y0' in condizioni:
                    ics[yf(var_sym).diff(var_sym, 3).subs(var_sym, x0_val)] = condizioni['d3y0']
                sol_part = dsolve(eq, y_sym, ics=ics).rhs
            except Exception:
                try:
                    sol_part = _cauchy_manuale(sol_gen, condizioni, var_sym)
                except Exception: pass
                
        latex_steps = []
        latex_steps.append(_step("Input ricevuto", latex_input))
        if latex_exp:
            latex_steps.append(_step("Equazione espansa", latex_exp))
        new_steps, sol_gen, sol_part = genera_passaggi(eq, tipo_key, ordine, var_sym, sol_gen, sol_part, condizioni)
        latex_steps += new_steps
        
        calc_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "success": True, 
            "latex": latex_steps,
            "ordine": ordine,
            "tipo": tipo_key,
            "nome_tipo": nome_tipo,
            "variabile": str(var_sym),
            "soluzione_generale": latex(_clean_ode_sol(sol_gen, var_sym)),
            "soluzione_particolare": latex(_clean_ode_sol(sol_part, var_sym)) if sol_part else None,
            "tempo_calcolo": calc_time
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def _cauchy_manuale(sol_gen, cond, var_sym):
    x0 = cond['x0']
    consts = _get_constants(sol_gen)
    if not consts:
        consts = [s for s in sol_gen.free_symbols 
                  if s != var_sym and not str(s).startswith('e') and not str(s).startswith('pi')]
    
    eqs = []
    if 'y0' in cond:
        eqs.append(Eq(sol_gen.subs(var_sym, x0), cond['y0']))
    if 'dy0' in cond:
        eqs.append(Eq(diff(sol_gen, var_sym).subs(var_sym, x0), cond['dy0']))
    if 'd2y0' in cond:
        eqs.append(Eq(diff(sol_gen, var_sym, 2).subs(var_sym, x0), cond['d2y0']))
    if 'd3y0' in cond:
        eqs.append(Eq(diff(sol_gen, var_sym, 3).subs(var_sym, x0), cond['d3y0']))
    
    if len(eqs) != len(consts):
        return None
    
    sol = solve(eqs, consts, dict=True)
    if sol:
        return _clean_ode_sol(sol_gen.subs(sol[0]), var_sym)
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
    use_generic = data.get("generic_conditions", True)

    if not equazione:
        return jsonify({"success": False, "error": "Nessuna equazione fornita."})

    condizioni = {}
    if not use_generic:
        try:
            x0_str = data.get("x0", "").strip()
            y0_str = data.get("y0", "").strip()
            dy0_str = data.get("dy0", "").strip()
            d2y0_str = data.get("d2y0", "").strip()
            d3y0_str = data.get("d3y0", "").strip()

            if x0_str and y0_str:
                condizioni['x0'] = _parse_valore(x0_str)
                condizioni['y0'] = _parse_valore(y0_str)
                if dy0_str:
                    condizioni['dy0'] = _parse_valore(dy0_str)
                if d2y0_str:
                    condizioni['d2y0'] = _parse_valore(d2y0_str)
                if d3y0_str:
                    condizioni['d3y0'] = _parse_valore(d3y0_str)
            else:
                # Se l'utente ha deselezionato 'generiche' ma non ha messo x0/y0
                condizioni = None
        except Exception as e:
            return jsonify({"success": False, "error": f"Errore condizioni iniziali: {e}"})
    else:
        condizioni = None

    return jsonify(risolvi_equazione_differenziale(equazione, condizioni))

# ═══════════════════ WRAPPER RETROCOMPATIBILE ═══════════════════
def solve_differential_equation(equation_str, conditions=None):
    """Compatibilità con condizioni_differenziali.py."""
    try:
        eq = equation_str.strip()
        eq = eq.replace('\\Delta', 'Δ')
        # La normalizzazione x/t è ora gestita internamente da risolvi_equazione_differenziale -> parsifica_input
        return risolvi_equazione_differenziale(eq, conditions)
    except Exception as e:
        return {"success": False, "error": str(e)}
