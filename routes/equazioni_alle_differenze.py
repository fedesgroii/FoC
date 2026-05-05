"""
routes/equazioni_alle_differenze.py – Risolutore di equazioni alle differenze (recurrence relations).
Supporta shift y(n+1), operatore differenza Δ e operatore shift E.
Copia la logica di equazioni_differenziali.py.
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

def _normalize_constants(expr):
    """Normalizza i nomi delle costanti in C_1, C_2, ... in ordine alfabetico."""
    if expr is None:
        return None
    c_map = {}
    found = _get_constants(expr)
    for i, c in enumerate(found):
        c_map[c] = symbols(f'C_{i+1}')
    return simplify(expr.subs(c_map))

# ═══════════════════ VARIABILE INDIPENDENTE ═══════════════════
def _detect_var(testo):
    """Rileva automaticamente la variabile indipendente (t, n, k o x)."""
    clean = re.sub(r'(sin|cos|tan|sqrt|exp|text|latex|log|ln)\b', '', testo)
    if re.search(r'\bt\b', clean):
        return 't'
    if re.search(r'\bn\b', clean):
        return 'n'
    if re.search(r'\bk\b', clean):
        return 'k'
    return 'n'

def _normalize_variables(testo, var_target):
    """Normalizza le variabili indipendenti nella variabile target."""
    vars_pos = ['n', 't', 'k', 'x']
    for v in vars_pos:
        if v != var_target:
            testo = re.sub(rf'\b{v}\b', var_target, testo)
    return testo

def _get_symbols(var_name):
    v = symbols(var_name, integer=True)
    yf = Function('y')
    return v, yf, yf(v)

def _normalize(testo):
    s = testo.strip()
    s = s.replace('\\Delta', 'Δ')
    s = s.replace('∘', ' ')
    s = s.replace('·', '*')
    
    # Rimuove spazi intorno all'esponente
    s = re.sub(r'\s*\^\s*\(', '^(', s)
    
    # FIX: Gestione specifica operatori d, Δ, E con potenze numeriche e simboliche
    s = re.sub(r'\b([dΔ])\s*\^?\s*\(?(\d+|[nkxtm])\)?', r'(\1)**\2', s)
    s = re.sub(r'\bE\s*\^?\s*\(?(\d+|[nkxtm])\)?', r'(E)**\2', s)
    
    # Sostituzione residua ^ -> **
    s = s.replace('^', '**')
    
    # d residui (senza potenza) diventano Δ
    s = s.replace('d', 'Δ')
    
    # Unicode superscript
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
    
    # Aggiunge y mancante dopo un operatore (Δ o E)
    if 'y' not in s:
        if '=' in s:
            lhs, rhs = s.split('=', 1)
            lhs = lhs.strip()
            if lhs and (lhs[-1].isdigit() or lhs[-1] == ')' or lhs[-1] in ('Δ', 'E', 'd')):
                lhs += '*y'
            s = f"{lhs} = {rhs}"
        else:
            s += '*y'
            
    return s

def _is_operator_form(lhs):
    if "y(" in lhs:
        return False
    if re.search(r'[ΔE]', lhs):
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
    
    # FIX: Controllo esplicito se il polinomio è vuoto
    if degree < 0:
        raise ValueError("L'operatore non contiene E: impossibile espandere in shift.")
    
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
    # FIX: Spostata la sostituzione di ** -> ^ dopo le altre elaborazioni
    s = s.replace('Δ', r'\Delta ')
    s = s.replace('∘', r'\circ ')
    s = s.replace('·', r'\cdot ')
    s = s.replace('*', r'\cdot ')
    
    s = s.replace('⁰', '^{0}').replace('¹', '^{1}').replace('²', '^{2}').replace('³', '^{3}')
    s = s.replace('⁴', '^{4}').replace('⁵', '^{5}').replace('⁶', '^{6}').replace('⁷', '^{7}')
    s = s.replace('⁸', '^{8}').replace('⁹', '^{9}')
    
    # FIX: Conversione ** -> ^ solo alla fine, dopo aver gestito tutto il resto
    s = re.sub(r'([a-zA-Z0-9)\]])\*\*([a-zA-Z0-9(])', r'\1^\2', s)
    
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
    lhs_expr = _parse_side(lhs_str.strip(), var_sym)
    rhs_expr = _parse_side(rhs_str.strip(), var_sym)
    eq = Eq(lhs_expr, rhs_expr)
    
    shifts = []
    def find_shifts(expr):
        if isinstance(expr, Function('y')):
            arg = expr.args[0]
            if arg == var_sym: shifts.append(0)
            else:
                diff = simplify(arg - var_sym)
                if diff.is_number: shifts.append(int(diff))
        elif hasattr(expr, 'args'):
            for a in expr.args: find_shifts(a)
            
    find_shifts(eq.lhs - eq.rhs)
    if not shifts: ordine = 1
    else: ordine = max(shifts) - min(shifts)
    
    return eq, max(1, ordine), latex(eq)

def _parse_side(s, var_sym):
    if not s.strip(): return sp.Integer(0)
    var_name = str(var_sym)
    yf = Function('y')
    
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
    
    testo_orig = _normalize_variables(testo_orig, var_name)
    testo_norm = _normalize(testo_orig)
    _log(f"Normalizzato: {testo_norm}")
    
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
def classifica_eq(eq, y_sym, ordine, var_sym):
    eq_exp = expand(eq.lhs - eq.rhs)
    is_constant = True
    for term in (eq_exp.args if isinstance(eq_exp, sp.Add) else [eq_exp]):
        if term.has(Function('y')):
            coeff = term
            for f in (term.args if isinstance(term, sp.Mul) else [term]):
                if f.has(Function('y')):
                    coeff = term / f
                    break
            if coeff.has(var_sym):
                is_constant = False
                break
    
    tipo_key = f"{ordine}th_linear"
    if is_constant: tipo_key += "_constant_coeff"
    if eq.rhs == 0: tipo_key += "_homogeneous"
    else: tipo_key += "_nonhomogeneous"
    
    nome = f"Equazione alle differenze lineare di ordine {ordine}"
    if is_constant: nome += " a coefficienti costanti"
    else: nome += " a coefficienti variabili"
    
    return tipo_key, nome

# ═══════════════════ PASSAGGI RISOLUTIVI ═══════════════════
def _step(title, content):
    return {"title": title, "content": content}

def _get_constants(expr):
    """Estrae le costanti di integrazione (C0, C1, C_1, C₁, ecc.) ordinate per indice."""
    consts = []
    for s in expr.free_symbols:
        name = str(s)
        # FIX: Supporto per pedici unicode (C₁, C₂, ...)
        name_clean = name.replace('₀','0').replace('₁','1').replace('₂','2').replace('₃','3').replace('₄','4').replace('₅','5').replace('₆','6').replace('₇','7').replace('₈','8').replace('₉','9')
        match = re.match(r'^C_?(\d+)$', name_clean)
        if match:
            consts.append((s, int(match.group(1))))
    return [c[0] for c in sorted(consts, key=lambda x: x[1])]

def _solve_for_constants_generic(sol_complete, order, var_sym):
    """Calcola le costanti in funzione di y0, y1, ... con passaggi espliciti su più righe."""
    # FIX: Usa _get_constants unificata invece di regex duplicata
    consts = _get_constants(sol_complete)
    if len(consts) < order:
        _log(f"Fallimento: trovate solo {len(consts)} costanti per ordine {order}. Costanti rilevate: {consts}")
        return {}, []
    
    steps = []
    y_syms = [symbols(f'y_{i}') for i in range(order)]
    eqs = []
    
    syst_latex = []
    for i in range(order):
        val_i = simplify(sol_complete.subs(var_sym, i).expand())
        eqs.append(Eq(val_i, y_syms[i]))
        syst_latex.append(rf"y({i}) = {latex(val_i)} = y_{{{i}}}")
    
    steps.append(_step("Sistema per le condizioni iniziali", 
        r" \begin{cases} " + r" \\ ".join(syst_latex) + r" \end{cases} "))
    
    sol = solve(eqs, consts, dict=True)
    if not sol: 
        _log(f"Fallimento: il sistema per le condizioni iniziali non ha soluzione. Equazioni: {eqs}")
        return {}, steps
    
    sol_dict = {k: simplify(v) for k, v in sol[0].items()}
    sol_latex_lines = [f"{latex(c)} = {latex(sol_dict[c])}" for c in consts if c in sol_dict]
    sol_latex = r" \\ ".join(sol_latex_lines)
    
    steps.append(_step("Costanti determinate", 
        r"\text{Risolvendo il sistema:}" + r" \\ " + sol_latex))
    
    return sol_dict, steps

def genera_passaggi(eq, tipo_key, ordine, var_sym, total_sol, sol_part_cauchy=None, cond=None):
    """Genera i passaggi seguendo i 6 step pedagogici."""
    yf = Function('y')
    y_sym = yf(var_sym)
    v = str(var_sym)
    steps = []

    _, nome_tipo = classifica_eq(eq, y_sym, ordine, var_sym)
    steps.append(_step("Classificazione", rf"\text{{{nome_tipo}}}"))

    # ── Separazione robusta y_h / y_p ────────────────────────────────────────
    consts_in_sol = _get_constants(total_sol)
    if eq.rhs == 0:
        y_h = total_sol
        y_p = sp.Integer(0)
    elif consts_in_sol:
        y_p = simplify(total_sol.subs({c: 0 for c in consts_in_sol}))
        y_h = simplify(total_sol - y_p)
    else:
        _log("Nessuna costante trovata in total_sol.")
        y_h = sp.Integer(0)
        y_p = total_sol

    if 'constant_coeff' in tipo_key:
        # Passo 2: soluzione omogenea con radici
        new_steps, _ = _steps_coeff_costanti(eq, ordine, var_sym, y_sym, total_sol)
        steps += new_steps

        # Passo 3: soluzione particolare
        if eq.rhs != 0:
            _, radici_dict = _build_char_poly(eq, ordine, var_sym)
            yp_analitica, steps_yp, is_resonant = _trova_yp_analitica(eq, var_sym, radici_dict)
            steps += steps_yp
            if not is_resonant and yp_analitica is not None:
                y_p_display = yp_analitica
            else:
                y_p_display = y_p
            if y_p_display != 0:
                steps.append(_step("Passo 3 – Soluzione particolare verificata",
                    rf"y_p({v}) = {latex(simplify(y_p_display))}"))
        else:
            y_p_display = sp.Integer(0)
    else:
        steps.append(_step("Metodo", r"\text{Risoluzione simbolica con SymPy (coeff. variabili)}"))
        y_p_display = y_p

    # Passo 4: soluzione generale
    sol_gen_esplicita = simplify(y_h + y_p)
    steps.append(_step("Passo 4 – Soluzione generale",
        rf"y_{{g,no}}({v}) = y_{{g,o}}({v}) + y_p({v}) "\
        rf"= \underbrace{{{latex(y_h)}}}_{{y_{{g,o}}}} + \underbrace{{{latex(y_p)}}}_{{y_p}}"))
    steps.append(_step("Soluzione generale (box)",
        rf"\boxed{{y_{{g,no}}({v}) = {latex(sol_gen_esplicita)}}}"))

    # Passo 5: condizioni iniziali
    if cond and sol_part_cauchy is not None:
        parts = [rf"y({i}) = {latex(cond[f'y{i}'])}" for i in range(ordine) if f'y{i}' in cond]
        steps.append(_step("Passo 5 – Condizioni iniziali", r", \quad ".join(parts)))
        steps.append(_step("Passo 5 – Soluzione del Problema di Cauchy",
            rf"\boxed{{y({v}) = {latex(simplify(sol_part_cauchy))}}}"))
    elif not cond:
        sol_dict, generic_steps = _solve_for_constants_generic(total_sol, ordine, var_sym)
        if sol_dict:
            steps.append(_step("Passo 5 – Imposizione condizioni iniziali generiche",
                rf"\text{{Si impone }} y(i) = y_i \text{{ per }} i = 0,\ldots,{ordine-1}"))
            steps.extend(generic_steps)
            sol_gen_sub = simplify(total_sol.subs(sol_dict))
            steps.append(_step("Passo 5 – Soluzione in funzione dei valori iniziali",
                rf"\boxed{{y({v}) = {latex(sol_gen_sub)}}}"))
            legenda = [rf"y_{{{i}}} = y({i})" for i in range(ordine)]
            steps.append(_step("Legenda", rf"\text{{Dove: }} {', '.join(legenda)}"))

    # Passo 6: risposta libera e forzata
    if cond and sol_part_cauchy is not None:
        # Risposta libera: soluzione con ingresso nullo
        try:
            eq_hom = Eq(eq.lhs, 0)
            sol_libera = rsolve(eq_hom, y_sym, {yf(i): cond[f'y{i}'] for i in range(ordine) if f'y{i}' in cond})
            sol_libera = simplify(sol_libera) if sol_libera else None
        except Exception:
            sol_libera = None
        # Risposta forzata: condizioni iniziali nulle
        try:
            ics_zero = {yf(i): sp.Integer(0) for i in range(ordine)}
            sol_forzata = rsolve(eq, y_sym, ics_zero)
            sol_forzata = simplify(sol_forzata) if sol_forzata else None
        except Exception:
            sol_forzata = None

        passo6_parts = []
        if sol_libera is not None:
            passo6_parts.append(rf"y_l({v}) = {latex(sol_libera)} \quad \text{{(risposta libera: condiz. iniz., ingresso nullo)}}")
        if sol_forzata is not None:
            passo6_parts.append(rf"y_f({v}) = {latex(sol_forzata)} \quad \text{{(risposta forzata: condiz. iniz. nulle)}}")
        if passo6_parts:
            steps.append(_step("Passo 6 – Risposta libera e forzata",
                r" \\ ".join(passo6_parts)))

    return steps, sol_gen_esplicita

def _build_char_poly(eq, ordine, var_sym):
    """Costruisce il polinomio caratteristico e restituisce (char_poly, radici_dict)."""
    r_sym = symbols('r')
    eq_expr = expand(eq.lhs - eq.rhs)
    shifts = []
    def find_shifts(expr):
        if isinstance(expr, Function('y')):
            arg = expr.args[0]
            if arg == var_sym: shifts.append(0)
            else:
                d = simplify(arg - var_sym)
                if d.is_number: shifts.append(int(d))
        elif hasattr(expr, 'args'):
            for a in expr.args: find_shifts(a)
    find_shifts(eq_expr)
    m = min(shifts) if shifts else 0
    coeffs = {}
    def get_c(expr):
        if isinstance(expr, sp.Add):
            for t in expr.args: get_c(t)
        else:
            c, r2 = expr.as_coeff_Mul()
            for f in (r2.args if isinstance(r2, sp.Mul) else [r2]):
                if isinstance(f, Function('y')):
                    s = int(simplify(f.args[0] - var_sym))
                    coeffs[s - m] = c * (expr / f / c)
    get_c(expand(eq.lhs))
    max_shift = max(coeffs.keys()) if coeffs else ordine
    char_poly = simplify(sum(coeffs.get(i, 0) * r_sym**i for i in range(max_shift + 1)))
    radici_dict = sp.roots(char_poly, r_sym)
    return char_poly, radici_dict


def _trova_yp_analitica(eq, var_sym, radici_dict):
    """Passo 3: soluzione particolare analitica con rilevamento risonanza.
    Restituisce (y_p, steps_yp, is_resonant)."""
    v = str(var_sym)
    u = eq.rhs
    steps_yp = []
    if u == 0:
        return sp.Integer(0), [], False

    # Riconosce u(t) = A * b^t
    b_val = None
    A_coeff = None
    try:
        # Prova a identificare b come base esponenziale dominante
        u_exp = expand(u)
        # Caso semplice: u = c * b^t
        if u_exp.is_Mul or u_exp.is_Pow or u_exp.is_Number or u_exp.is_Symbol:
            as_pow = u_exp.rewrite(sp.Pow)
            for atom in u_exp.atoms(sp.Pow):
                if atom.args[1] == var_sym:
                    b_val = atom.args[0]
                    A_coeff = simplify(u_exp / atom)
                    break
            if b_val is None and u_exp.is_Number:
                b_val = u_exp
                A_coeff = sp.Integer(1)
    except Exception:
        pass

    if b_val is not None:
        # Valuta p(b)
        r_sym = symbols('r')
        char_poly, _ = _build_char_poly(eq, 0, var_sym)
        p_b = char_poly.subs(r_sym, b_val)
        p_b_simplified = simplify(p_b)

        # Controlla risonanza: b è radice di p?
        mol = radici_dict.get(b_val, 0)
        if p_b_simplified != 0:
            # CASO SENZA RISONANZA: y_p = A/p(b) * b^t
            yp = simplify(A_coeff / p_b_simplified * b_val**var_sym)
            steps_yp.append(_step("Passo 3 – Soluzione Particolare (senza risonanza)",
                rf"u({v}) = {latex(u)}, \quad p(b) = p({latex(b_val)}) = {latex(p_b_simplified)} \neq 0 "\
                rf"\\ y_p({v}) = \frac{{1}}{{p(b)}} \cdot u({v}) = \frac{{1}}{{{latex(p_b_simplified)}}} \cdot {latex(b_val)}^{{{v}}} = {latex(yp)}"))
            return yp, steps_yp, False
        else:
            # CASO CON RISONANZA: moltiplica per t^m
            steps_yp.append(_step("Passo 3 – Risonanza rilevata",
                rf"b = {latex(b_val)} \text{{ è radice di }} p(\Delta) \text{{ con molteplicità }} m={mol}. "\
                rf"\text{{Si pone }} y_p({v}) = \alpha \cdot {v}^{{{mol}}} \cdot {latex(b_val)}^{{{v}}}"))
            alpha = symbols('alpha')
            yp_ansatz = alpha * var_sym**mol * b_val**var_sym
            return None, steps_yp, True  # segnala risonanza, usa rsolve

    # Ingresso polinomiale generico: usa rsolve direttamente
    steps_yp.append(_step("Passo 3 – Soluzione Particolare",
        rf"u({v}) = {latex(u)} \text{{: forma non esponenziale pura, si usa il metodo di variazione dei parametri.}}"))
    return None, steps_yp, False


def _steps_coeff_costanti(eq, ordine, var_sym, y_sym, sol_gen):
    """Passo 2: costruisce i passaggi per la soluzione omogenea con tabella dei modi."""
    steps = []
    v = str(var_sym)
    r_sym = symbols('r')

    char_poly, radici_dict = _build_char_poly(eq, ordine, var_sym)

    # — Passo 1: forma operatoriale —
    steps.append(_step("Passo 1 – Forma operatoriale",
        rf"p(\Delta)\,y({v}) = {latex(eq.rhs)} \quad \text{{dove }} p(\Delta) = {latex(char_poly.subs(r_sym, symbols('Delta')))}"))

    # — Passo 2a: polinomio caratteristico —
    steps.append(_step("Passo 2 – Polinomio caratteristico", rf"p(r) = {latex(char_poly)} = 0"))

    # — Passo 2b: radici e tabella modi —
    rad_lines = []
    for r_val, mol in radici_dict.items():
        if sp.im(r_val) == 0:
            if r_val == 0:
                modo = rf"c_1 \delta_0({v})+\ldots" if mol > 1 else rf"c\,\delta_0({v})"
            elif mol == 1:
                modo = rf"c \cdot ({latex(r_val)})^{{{v}}}"
            else:
                modo = rf"(c_1 + c_2 {v} + \ldots + c_{{{mol}}} {v}^{{{mol-1}}})\,({latex(r_val)})^{{{v}}}"
            tipo = "reale semplice" if mol == 1 else f"reale multipla (m={mol})"
        else:
            rho = simplify(Abs(r_val))
            theta = simplify(sp.arg(r_val))
            modo = rf"{latex(rho)}^{{{v}}}\,(c_1\cos({latex(theta)}\,{v}) + c_2\sin({latex(theta)}\,{v}))"
            tipo = "complessa coniugata"
        rad_lines.append(rf"r = {latex(r_val)} \;(\text{{{tipo}, molt.}} = {mol}) \;\Rightarrow\; {modo}")
    steps.append(_step("Passo 2 – Radici e modi", r" \\ ".join(rad_lines)))

    # — Stabilità —
    stab = "asintoticamente stabile"
    for r_val, mol in radici_dict.items():
        rho_ev = Abs(r_val).evalf()
        if rho_ev > 1: stab = "instabile"; break
        if rho_ev == 1:
            stab = "instabile" if mol > 1 else "marginalmente stabile"
    steps.append(_step("Stabilità",
        rf"\text{{Tutte le radici: }} |r_i| \Rightarrow \text{{sistema }} \mathbf{{{stab}}}. "\
        rf"\text{{(stabile sse }} |r_i| < 1 \;\forall i\text{{)}}"))

    # — Costruzione y_h in forma reale —
    termini = []
    c_idx = 1
    processed_complex = set()
    for r_val, mol in radici_dict.items():
        if sp.im(r_val) == 0:
            for k in range(mol):
                term = r_val**var_sym
                if k > 0: term *= var_sym**k
                termini.append(symbols(f'C_{c_idx}') * term)
                c_idx += 1
        else:
            if r_val in processed_complex or sp.conjugate(r_val) in processed_complex:
                continue
            rho = simplify(sp.Abs(r_val))
            theta = simplify(sp.arg(r_val))
            for k in range(mol):
                prefix = simplify(rho**var_sym)
                if k > 0: prefix *= var_sym**k
                c1 = symbols(f'C_{c_idx}')
                c2 = symbols(f'C_{c_idx+1}')
                termini.append(prefix * (c1 * cos(theta * var_sym) + c2 * sin(theta * var_sym)))
                c_idx += 2
            processed_complex.add(r_val)

    y_h = sp.Add(*termini) if termini else sp.Integer(0)
    steps.append(_step("Passo 2 – Soluzione omogenea",
        rf"y_{{g,o}}({v}) = {latex(simplify(y_h))}"))
    return steps, y_h

# ═══════════════════ RISOLUTORE PRINCIPALE ═══════════════════
def risolvi_eq_alle_differenze(input_utente, condizioni=None, use_generic=True):
    start_time = time.time()
    try:
        eq, ordine, var_sym, latex_input, latex_exp = parsifica_input(input_utente)
        yf = Function('y')
        y_sym = yf(var_sym)
        tipo_key, nome_tipo = classifica_eq(eq, y_sym, ordine, var_sym)
        
        ics = {}
        if condizioni:
            for i in range(ordine + 1):
                if f'y{i}' in condizioni:
                    ics[yf(i)] = condizioni[f'y{i}']
        
        soluzione_generale = None
        try:
            soluzione_generale = rsolve(eq, y_sym)
        except Exception as e:
            _log(f"rsolve generale fallito: {e}")
            
        if soluzione_generale is None: 
            raise ValueError("Impossibile trovare la soluzione generale dell'equazione.")
            
        sol_part_cauchy = None
        if ics:
            try:
                sol_part_cauchy = rsolve(eq, y_sym, ics)
            except Exception as e:
                _log(f"rsolve particolare fallito: {e}")
        
        total_sol = simplify(expand(soluzione_generale))

        total_sol = _normalize_constants(total_sol)
        sol_part_cauchy = _normalize_constants(sol_part_cauchy)

        latex_steps = [_step("Input ricevuto", latex_input)]
        if latex_exp: latex_steps.append(_step("Equazione espansa", latex_exp))
        
        new_steps, sol_gen_final = genera_passaggi(eq, tipo_key, ordine, var_sym, total_sol, sol_part_cauchy, condizioni if not use_generic else None)
        latex_steps += new_steps
        
        stab_str = "asintoticamente stabile"
        try:
            r_sym = symbols('r')
            eq_h = expand(eq.lhs)
            coeffs = {}
            def get_c(expr):
                if isinstance(expr, sp.Add):
                    for t in expr.args: get_c(t)
                else:
                    c, r = expr.as_coeff_Mul()
                    for f in (r.args if isinstance(r, sp.Mul) else [r]):
                        if isinstance(f, Function('y')):
                            s = int(simplify(f.args[0] - var_sym))
                            coeffs[s] = c * (expr / f / c)
            get_c(eq_h)
            m = min(coeffs.keys())
            cp = sum(coeffs.get(i, 0) * r_sym**(i-m) for i in coeffs)
            roots = sp.roots(cp, r_sym)
            for r, mol in roots.items():
                rho = Abs(r).evalf()
                if rho > 1: stab_str = "instabile"; break
                if rho == 1:
                    if mol > 1: stab_str = "instabile"; break
                    stab_str = "marginalmente stabile"
        except: stab_str = "non determinata"

        calc_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "success": True,
            "latex": latex_steps,
            "ordine": ordine,
            "tipo": tipo_key,
            "nome_tipo": nome_tipo,
            "variabile": str(var_sym),
            "soluzione_generale": latex(simplify(sol_gen_final)),
            "soluzione_particolare": latex(simplify(sol_part_cauchy)) if sol_part_cauchy else None,
            "stabilita": stab_str,
            "tempo_calcolo": calc_time
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def _parse_valore(s):
    s = s.strip().replace('^', '**')
    return parse_expr(s, local_dict={'e': sp.E, 'pi': pi, 'Rational': Rational}, transformations=_transformations)

@equazioni_alle_differenze_bp.route('/api/equazioni_alle_differenze', methods=['POST'])
def api_eq_alle_differenze():
    data = request.get_json()
    eq = data.get("equazione", "").strip()
    if not eq: return jsonify({"success": False, "error": "Nessuna equazione fornita."})
    
    use_generic = data.get("generic_conditions", True)
    condizioni = {}
    try:
        for i in range(5):
            val = data.get(f"y{i}", "").strip()
            if val: condizioni[f"y{i}"] = _parse_valore(val)
    except Exception as e:
        return jsonify({"success": False, "error": f"Errore condizioni: {e}"})
        
    return jsonify(risolvi_eq_alle_differenze(eq, condizioni or None, use_generic))