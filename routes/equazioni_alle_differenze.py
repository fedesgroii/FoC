"""
routes/equazioni_alle_differenze.py – Risolutore di equazioni alle differenze (recurrence relations).
Supporta shift y(n+1), operatore differenza Δ e operatore shift E.
Copia la logica di equazioni_differenziali.py.
"""
from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import (symbols, Function, Eq, rsolve, exp, simplify,
    cos, sin, tan, ln, log, latex, expand, solve, I, pi, oo,
    sqrt, Rational, Abs, diff, Poly, KroneckerDelta, arg)
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
    s = re.sub(r'\bE\s*\^?\s*\(?(\d+|[nkxtm])\)?', r'(E)**\1', s)
    
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
    
    # Se l'utente scrive y(t) lo lasciamo, ma se scrive y_t lo abbiamo già corretto.
    # Assicuriamoci che y(t) sia visto come funzione in seguito.
    
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
    # Se contiene Δ o E è probabilmente in forma operatoriale
    if re.search(r'[ΔE]', lhs):
        return True
    return False

def _prepare_op_str(s):
    s = s.strip()
    # Rimuove eventuali y(...), y o *y alla fine
    s = re.sub(r'[* ]*y(\([a-z]\))?$', '', s).strip()
    s = s.rstrip('* ')
    # Sostituisci Δ con E (nuova logica: Δ = shift puro)
    s = s.replace('Δ', 'E')
    # Gestisci potenze: E^n -> E**n
    s = re.sub(r'E\s*\^\s*(\d+)', r'E**\1', s)
    # Gestisci E2, E3 -> E**2, E**3
    s = re.sub(r'E\s*(\d+)', r'E**\1', s)
    if not s: return 'E'
    s = re.sub(r'\)\s*\(', ')*(', s)
    s = re.sub(r'\bE\s*\(', r'E*(', s)
    s = re.sub(r'\)\s*E\b', r')*E', s)
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)
    s = re.sub(r'\)\s*(\d)', r')*\1', s)
    return s

def _expand_operator(lhs_str, rhs_str, var_sym):
    op_str = _prepare_op_str(lhs_str)
    
    E_sym = symbols('E')
    # Troviamo tutti i simboli nell'operatore oltre a E
    unknown_syms = set(re.findall(r'\b([a-zA-Z])\b', op_str)) - {'E', 'e', 'i', 'j'}
    local = {'E': E_sym, 'e': sp.E, 'pi': pi, 'Rational': Rational}
    for s in unknown_syms: local[s] = symbols(s)
    
    try:
        op_expr = parse_expr(op_str, local_dict=local, transformations=_transformations)
    except Exception as e:
        raise ValueError(f"Impossibile interpretare l'operatore '{op_str}': {e}")
        
    op_expanded = expand(op_expr)
    
    # Utilizziamo Poly per estrarre i coefficienti di E**k in modo robusto
    # Ogni E**k corrisponde direttamente a y(t+k) secondo la nuova logica Δ = E
    try:
        poly = Poly(op_expanded, E_sym)
        coeffs_dict = poly.as_dict()
    except:
        # Se non è un polinomio in E (es. costante), lo trattiamo come termine in y(t)
        coeffs_dict = {(0,): op_expanded}

    yf = Function('y')
    lhs_eq = sp.Integer(0)
    for monom, coeff in coeffs_dict.items():
        n_shift = monom[0]
        lhs_eq += coeff * yf(var_sym + n_shift)
        
    rhs_parsed = _parse_rhs(rhs_str, var_sym)
    eq = Eq(lhs_eq, rhs_parsed)
    
    # Calcolo ordine basato sugli shift presenti
    y_calls = lhs_eq.atoms(Function)
    shifts = []
    for yc in y_calls:
        if str(yc.func) == 'y':
            d = simplify(yc.args[0] - var_sym)
            if d.is_number: shifts.append(int(d))
    
    ordine = max(shifts) - min(shifts) if shifts else 1
    
    return eq, max(1, ordine), latex(eq)

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
    for yc in eq.lhs.atoms(Function):
        if str(yc.func) == 'y':
            d = simplify(yc.args[0] - var_sym)
            if d.is_number:
                shifts.append(int(d))
            
    if not shifts: ordine = 1
    else: ordine = max(shifts) - min(shifts)
    
    return eq, max(1, ordine), latex(eq)

def _parse_side(s, var_sym):
    if not s.strip(): return sp.Integer(0)
    var_name = str(var_sym)
    yf = Function('y')
    
    # Trova simboli sconosciuti
    unknown_syms = set(re.findall(r'\b([a-z])\b', s)) - {var_name, 'e', 'i', 'j'}
    local = {var_name: var_sym, 'e': sp.E, 'E': sp.E, 'pi': pi, 'exp': exp,
             'sin': sin, 'cos': cos, 'tan': tan, 'ln': ln, 'log': log,
             'sqrt': sqrt, 'Rational': Rational, 'I': I,
             '__YFUNC__': yf}
    for sym in unknown_syms: local[sym] = symbols(sym)
    
    # Sostituiamo y(...) con __YFUNC__(...) SENZA word boundary prima di y
    s = re.sub(r'(?<![a-zA-Z_])y\(([^)]+)\)', r'__YFUNC__(\1)', s)
    s = re.sub(r'(?<![a-zA-Z_\d])(\d+)\s*/\s*(\d+)(?![a-zA-Z_\d\(])', r'Rational(\1,\2)', s)
    s = re.sub(r'(\d)(__YFUNC__)', r'\1*\2', s)
    s = re.sub(r'(' + var_name + r')(__YFUNC__)', r'\1*\2', s)
    s = re.sub(r'\)(__YFUNC__)', r')*\1', s)
    
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
        # Messaggio informativo sulla definizione di Δ
        info_msg = r"\text{Nota: l'operatore } \Delta \text{ viene interpretato come shift: } \Delta^n y(t) = y(t+n)"
        return eq, ordine, var_sym, latex_sanitized + r" \\ " + info_msg, latex_exp
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
    return {"title": title, "content": rf"\text{{{title}}}" + r" \\ " + content}

def _boxed_large(content, color="yellow", size="Large"):
    """
    Genera un box colorato con testo grande per MathJax/KaTeX.
    """
    color_map = {
        "yellow": "#FFFFCC", # Giallo chiaro
        "blue": "#E6F3FF",   # Azzurro chiaro
        "green": "#E6FFE6",  # Verde chiaro
        "cyan": "#E6FFFF"
    }
    bg = color_map.get(color, "#FFFFCC")
    # Utilizziamo \bbox che è universalmente supportato da MathJax/KaTeX per sfondi colorati
    return rf"\bbox[{bg}, 10pt, border:1px solid #CCCCCC]{{\{size} {content}}}"

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

def _solve_for_constants_robust(sol_gen, order, var_sym, condizioni=None):
    """Calcola le costanti (C1, C2...) in modo robusto (numerico o generico)."""
    consts = _get_constants(sol_gen)
    if not consts: 
        return {}, [_step("Calcolo Costanti", r"\text{Nessuna costante da determinare.}")]
    
    steps = []
    eqs = []
    syst_latex = []
    
    # Se non ci sono condizioni numeriche, usiamo y_0, y_1...
    for i in range(order):
        key = f'y{i}'
        val_target = symbols(f'y_{i}')
        if condizioni and key in condizioni:
            val_target = condizioni[key]
            
        val_expr = simplify(sol_gen.subs(var_sym, i))
        eqs.append(Eq(val_expr, val_target))
        syst_latex.append(rf"y({i}) = {latex(val_expr)} = {latex(val_target)}")
    
    steps.append(_step("Sistema per le condizioni iniziali", 
                      r"\begin{aligned} " + r" \\ ".join(syst_latex) + r" \end{aligned}"))
    
    try:
        sol = solve(eqs, consts)
        if not sol:
            # Prova a risolvere singolarmente se il sistema globale fallisce
            sol = solve(eqs, consts, dict=True)
            if sol: sol = sol[0]
            else: return {}, steps + [_step("Errore", r"\text{Sistema impossibile o indeterminato.}")]
        
        if isinstance(sol, list): sol = sol[0]
        
        sol_dict = {k: simplify(v) for k, v in sol.items()}
        sol_latex = r" \\ ".join([f"{latex(c)} = {latex(sol_dict[c])}" for c in consts if c in sol_dict])
        steps.append(_step("Costanti determinate", r"\begin{aligned} " + sol_latex + r"\end{aligned}"))
        
        return sol_dict, steps
    except Exception as e:
        return {}, steps + [_step("Errore nel calcolo costanti", rf"\text{{{str(e)}}}")]

def genera_passaggi(eq, tipo_key, ordine, var_sym, condizioni=None, use_generic=True):
    """Genera i passaggi seguendo i 6 step pedagogici richiesti."""
    v = str(var_sym)
    yf = Function('y')
    steps = []

    # --- STEP 1: Analisi Operatore ---
    char_poly, radici_dict = _build_char_poly(eq, var_sym)
    r_sym = symbols('r')
    op_delta = char_poly.subs(r_sym, symbols('Delta'))
    steps.append(_step("Passo 1 – Analisi operatore e Polinomio Caratteristico",
        rf"P(\Delta) y({v}) = f({v}) \\"
        rf"P(\Delta) = {latex(op_delta)} \\"
        rf"\text{{Polinomio caratteristico: }} P(r) = {latex(char_poly)} = 0"))

    # --- STEP 2: Soluzione Omogenea (Tabella 1) ---
    y_h, steps_h = _calcola_omogenea_manuale(radici_dict, var_sym)
    steps.extend(steps_h)

    # --- STEP 3: Soluzione Particolare (Tabella 2) ---
    y_p, steps_p = _calcola_particolare_manuale(eq, var_sym, radici_dict)
    steps.extend(steps_p)

    # --- STEP 4: Soluzione Generale ---
    sol_gen = simplify(y_h + y_p)
    steps.append(_step("Passo 4 – Soluzione Generale",
        _boxed_large(rf"y_{{g,no}}({v}) = y_{{g,o}}({v}) + y_p({v}) = {latex(sol_gen)}", color="blue", size="Large")))

    # --- STEP 5: Calcolo Costanti (Sempre) ---
    sol_dict, steps_ci = _solve_for_constants_robust(sol_gen, ordine, var_sym, condizioni)
    steps.extend(steps_ci)
    sol_totale = simplify(sol_gen.subs(sol_dict))
    steps.append(_step("Soluzione Finale (Problema di Cauchy)",
        _boxed_large(rf"y({v}) = {latex(sol_totale)}", color="green", size="Huge")))

    # --- STEP 6: Risposta Libera e Forzata (Decomposizione) ---
    steps_lf = _calcola_libera_forzata_pedagogica(eq, ordine, var_sym, condizioni)
    steps.extend(steps_lf)

    return steps, sol_gen, sol_totale

def _calcola_omogenea_manuale(radici_dict, var_sym):
    """Implementa la Tabella 1: Radici -> Modi."""
    v = var_sym
    termini = []
    c_idx = 1
    rad_lines = []
    processed_complex = set()

    # Prova a ordinare le radici, altrimenti usa l'ordine originale
    try:
        sorted_roots = sorted(radici_dict.items(), key=lambda x: (not x[0].is_real, abs(x[0]) if x[0].is_number else 0))
    except:
        sorted_roots = radici_dict.items()

    for r_val, mol in sorted_roots:
        if r_val == 0:
            # Radice nulla: Kronecker delta
            tipo = "nulla semplice" if mol == 1 else f"nulla multipla (m={mol})"
            modi_r = []
            for k in range(mol):
                term = KroneckerDelta(v, k)
                const = symbols(f'C_{c_idx}')
                termini.append(const * term)
                modi_r.append(latex(term))
                c_idx += 1
            rad_lines.append(rf"r = 0 \;(\text{{{tipo}}}) \;\Rightarrow\; {', '.join(modi_r)}")
        elif r_val.is_real:
            # Radice reale
            tipo = "reale semplice" if mol == 1 else f"reale multipla (m={mol})"
            modi_r = []
            for k in range(mol):
                term = r_val**v
                if k > 0: term *= v**k
                const = symbols(f'C_{c_idx}')
                termini.append(const * term)
                modi_r.append(latex(term))
                c_idx += 1
            rad_lines.append(rf"r = {latex(r_val)} \;(\text{{{tipo}}}) \;\Rightarrow\; {', '.join(modi_r)}")
        else:
            # Radice complessa
            if r_val in processed_complex or sp.conjugate(r_val) in processed_complex:
                continue
            processed_complex.add(r_val)
            tipo = "complesse coniugate" if mol == 1 else f"complesse multiple (m={mol})"
            
            rho = simplify(Abs(r_val))
            theta = simplify(arg(r_val))
            
            modi_r = []
            for k in range(mol):
                prefix = rho**v
                if k > 0: prefix *= v**k
                
                c1 = symbols(f'C_{c_idx}')
                c2 = symbols(f'C_{c_idx+1}')
                termini.append(prefix * (c1 * cos(theta * v) + c2 * sin(theta * v)))
                
                modi_r.append(rf"{latex(prefix)}\cos({latex(theta)}{v}), {latex(prefix)}\sin({latex(theta)}{v})")
                c_idx += 2
            rad_lines.append(rf"r = {latex(r_val)} \pm {latex(sp.im(r_val))}j \;(\text{{{tipo}}}) \;\Rightarrow\; {', '.join(modi_r)}")

    y_h = sp.Add(*termini) if termini else sp.Integer(0)
    steps = [
        _step("Passo 2 – Radici e Modi (Tabella 1)", r" \\ ".join(rad_lines)),
        _step("Passo 2 – Soluzione Omogenea",
            _boxed_large(rf"y_{{g,o}}({v}) = {latex(y_h)}", color="yellow", size="Large"))
    ]
    return y_h, steps

def _calcola_particolare_manuale(eq, var_sym, radici_dict):
    """Implementa la Tabella 2: Metodo della Somiglianza."""
    v = var_sym
    f_t = simplify(eq.rhs)
    if f_t == 0:
        return sp.Integer(0), [_step("Passo 3 – Soluzione Particolare", r"f(t)=0 \Rightarrow y_p(t)=0")]

    # 1. Analisi dell'ingresso e scelta ansatz
    ansatz, coeffs, info_ansatz = _get_ansatz(f_t, v)
    
    # 2. Risonanza
    # Se l'ingresso ha una "base" r che è radice del pol. caratt.
    m = _check_resonance(f_t, v, radici_dict)
    if m > 0:
        ansatz = simplify(ansatz * v**m)
        info_ansatz += rf" \\ \text{{Risonanza rilevata: }} f(t) \text{{ ha termini in comune con }} y_h \text{{ (molteplicità }} m={m}\text{{). Moltiplico per }} {v}^{m}."

    steps = [_step("Passo 3 – Analisi Ingresso (Tabella 2)", 
                  rf"f({v}) = {latex(f_t)} \\ \text{{Ipotesi particolare: }} y_p({v}) = {latex(ansatz)} \\ {info_ansatz}")]

    # 3. Sostituzione nell'equazione per trovare i coefficienti
    lhs_expr = eq.lhs
    sub_map = {}
    
    def find_y_calls(expr):
        calls = set()
        if isinstance(expr, Function('y')):
            calls.add(expr)
        elif hasattr(expr, 'args'):
            for a in expr.args:
                calls.update(find_y_calls(a))
        return calls

    y_calls = find_y_calls(lhs_expr)
    for call in y_calls:
        arg_call = call.args[0]
        shift = simplify(arg_call - v)
        sub_map[call] = ansatz.subs(v, v + shift)

    substituted_lhs = expand(lhs_expr.subs(sub_map))
    
    sol_coeffs = solve_undetermined_coefficients(substituted_lhs, f_t, v, coeffs)
    
    if not sol_coeffs:
        try:
            yp_res = rsolve(eq, Function('y')(v))
            consts = _get_constants(yp_res)
            y_p_final = simplify(yp_res.subs({c: 0 for c in consts}))
            steps.append(_step("Passo 3 – Determinazione coefficienti", 
                              rf"\text{{Risolvendo per i coefficienti: }} y_p({v}) = {latex(y_p_final)}"))
            return y_p_final, steps
        except:
            return sp.Integer(0), steps + [_step("Errore", "Impossibile trovare soluzione particolare.")]

    y_p_final = simplify(ansatz.subs(sol_coeffs))
    
    coeff_lines = [f"{latex(c)} = {latex(val)}" for c, val in sol_coeffs.items()]
    steps.append(_step("Passo 3 – Determinazione coefficienti", 
                      rf"\text{{Sostituendo nell'equazione e uguagliando i termini:}} \\ " + 
                      r" \\ ".join(coeff_lines) + 
                      rf" \\ \Rightarrow y_p({v}) = {latex(y_p_final)}"))
    
    return y_p_final, steps

def _get_ansatz(f_t, v):
    """Restituisce (ansatz, list_of_coeffs, info_text)."""
    if f_t.has(KroneckerDelta):
        A = symbols('A')
        return A * KroneckerDelta(v, 0), [A], r"\text{Ingresso impulsivo } \delta_0(t)"

    if f_t.is_polynomial(v):
        deg = Poly(f_t, v).degree()
        coeffs = [symbols(f'A_{i}') for i in range(deg + 1)]
        ansatz = sum(coeffs[i] * v**i for i in range(deg + 1))
        return ansatz, coeffs, rf"\text{{Ingresso polinomiale di grado }} {deg}"

    b_found = None
    for atom in f_t.atoms(sp.Pow):
        if atom.args[1] == v:
            b_found = atom.args[0]
            break
    
    if b_found is not None:
        if f_t.has(sin) or f_t.has(cos):
            omega = 1
            for s in f_t.atoms(sin, cos):
                arg_s = s.args[0]
                omega = simplify(arg_s / v)
                break
            A, B = symbols('A B')
            ansatz = b_found**v * (A * cos(omega * v) + B * sin(omega * v))
            return ansatz, [A, B], rf"\text{{Ingresso sinusoidale smorzato: }} \rho = {latex(b_found)}, \omega = {latex(omega)}"
        else:
            A = symbols('A')
            return A * b_found**v, [A], rf"\text{{Ingresso esponenziale: }} b = {latex(b_found)}"

    if f_t.has(sin) or f_t.has(cos):
        omega = 1
        for s in f_t.atoms(sin, cos):
            arg_s = s.args[0]
            omega = simplify(arg_s / v)
            break
        A, B = symbols('A B')
        ansatz = A * cos(omega * v) + B * sin(omega * v)
        return ansatz, [A, B], rf"\text{{Ingresso sinusoidale: }} \omega = {latex(omega)}"

    A = symbols('A')
    return A * f_t, [A], r"\text{Ingresso generico (tentativo di somiglianza)}"

def _check_resonance(f_t, v, radici_dict):
    """Ritorna la molteplicità della risonanza."""
    b = 1
    for atom in f_t.atoms(sp.Pow):
        if atom.args[1] == v:
            b = atom.args[0]
            break
    
    omega = 0
    if f_t.has(sin) or f_t.has(cos):
        for s in f_t.atoms(sin, cos):
            omega = simplify(s.args[0] / v)
            break
            
    r_in = simplify(b * (cos(omega) + I * sin(omega)))
    
    for r_root, mol in radici_dict.items():
        if simplify(r_root - r_in) == 0 or simplify(r_root - sp.conjugate(r_in)) == 0:
            return mol
    return 0

def solve_undetermined_coefficients(lhs, rhs, v, coeffs):
    """Risolve il sistema per i coefficienti incogniti."""
    diff = expand(lhs - rhs)
    eqs = []
    for i in range(len(coeffs) + 2):
        eqs.append(diff.subs(v, i))
    
    sol = solve(eqs, coeffs)
    if isinstance(sol, list):
        return sol[0] if sol else {}
    return sol

def _calcola_libera_forzata_pedagogica(eq, ordine, var_sym, condizioni):
    """Calcola Risposta Libera (y_l) e Forzata (y_f) con logica decompositiva."""
    v = var_sym
    yf = Function('y')
    steps = []
    
    # 1. Condizioni iniziali (ICS) per rsolve
    ics = {}
    for i in range(ordine):
        key = f'y{i}'
        if condizioni and key in condizioni:
            ics[yf(i)] = condizioni[key]
        else:
            ics[yf(i)] = symbols(f'y_{i}')
            
    # 2. Risposta Libera (Omogenea + ICS)
    eq_hom = Eq(eq.lhs, 0)
    try:
        y_l = simplify(rsolve(eq_hom, yf(v), ics))
        y_l_display = y_l
    except:
        y_l_display = sp.Integer(0)

    # 3. Risposta Forzata (Completa + ICS nulle)
    ics_zero = {yf(i): 0 for i in range(ordine)}
    try:
        y_f = simplify(rsolve(eq, yf(v), ics_zero))
        y_f_display = y_f
    except:
        y_f_display = sp.Integer(0)

    steps.append(_step("Passo 6 – Risposta Libera e Forzata (Decomposizione)", 
                      rf"\begin{{aligned}} "
                      rf"y_l({v}) &= {latex(y_l_display)} \\ "
                      rf"y_f({v}) &= {latex(y_f_display)} \\ "
                      rf"\text{{Verifica: }} y_{{tot}} &= y_l + y_f = {latex(simplify(y_l_display + y_f_display))} "
                      rf"\end{{aligned}}"))
    return steps

def _build_char_poly(eq, var_sym):
    """Costruisce il polinomio caratteristico P(r) = 0 correttamente."""
    r_sym = symbols('r')
    # Porta tutto a sinistra per isolare i coefficienti
    eq_expanded = expand(eq.lhs - eq.rhs)
    
    # Trova tutti i termini y(t+k)
    yf_name = 'y'
    y_calls = [c for c in eq_expanded.atoms(Function) if str(c.func) == yf_name]
    
    coeffs = {}  # shift -> coefficiente
    for call in y_calls:
        arg = call.args[0]
        shift = simplify(arg - var_sym)
        if shift.is_number:
            shift_int = int(shift)
            # Estrai il coefficiente del termine y(t+shift)
            coeff = eq_expanded.coeff(call)
            coeffs[shift_int] = coeff
    
    if not coeffs:
        return sp.Integer(0), {}
    
    # Normalizza: shift minimo diventa 0 (forma standard)
    min_shift = min(coeffs.keys())
    char_poly = sp.Integer(0)
    for shift, coeff in coeffs.items():
        char_poly += coeff * r_sym**(shift - min_shift)
    
    # Trova radici del polinomio ridotto
    radici_dict = sp.roots(char_poly, r_sym)
    
    # Se lo shift minimo > 0, aggiungiamo radici in 0 (modi impulsivi)
    if min_shift > 0:
        radici_dict[sp.Integer(0)] = radici_dict.get(sp.Integer(0), 0) + min_shift
        # Aggiorniamo il polinomio per la visualizzazione (opzionale ma utile)
        char_poly = expand(char_poly * r_sym**min_shift)
    
    return char_poly, radici_dict

# ═══════════════════ RISOLUTORE PRINCIPALE ═══════════════════
def risolvi_eq_alle_differenze(input_utente, condizioni=None, use_generic=True):
    start_time = time.time()
    try:
        eq, ordine, var_sym, latex_input, latex_exp = parsifica_input(input_utente)
        yf = Function('y')
        y_sym = yf(var_sym)
        tipo_key, nome_tipo = classifica_eq(eq, y_sym, ordine, var_sym)
        
        latex_steps = [_step("Input ricevuto", latex_input)]
        
        # Legenda colori
        latex_steps.append(_step("Legenda colori",
            r"\text{📌 GIALLO = Soluzione Omogenea} \\" +
            r"\text{📌 AZZURRO = Soluzione Generale} \\" +
            r"\text{📌 VERDE = Soluzione Finale (con costanti)}"))

        if latex_exp: latex_steps.append(_step("Equazione espansa", latex_exp))
        
        # CHIAMATA AL NUOVO GENERATORE MANUALE
        new_steps, sol_gen_final, sol_part_cauchy = genera_passaggi(
            eq, tipo_key, ordine, var_sym, condizioni, use_generic
        )
        latex_steps += new_steps
        
        # Stabilità
        stab_str = "asintoticamente stabile"
        try:
            char_poly, radici_dict = _build_char_poly(eq, var_sym)
            for r, mol in radici_dict.items():
                rho = Abs(r)
                try:
                    if rho > 1: stab_str = "instabile"; break
                    if rho == 1:
                        if mol > 1: stab_str = "instabile"; break
                        stab_str = "marginalmente stabile"
                except (TypeError, ValueError):
                    stab_str = "non determinata"; break
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

@equazioni_alle_differenze_bp.route('/api/equazioni_alle_differenze/ordine', methods=['POST'])
def api_ordine():
    """Endpoint leggero per ottenere solo l'ordine dell'equazione."""
    data = request.get_json()
    eq = data.get("equazione", "").strip()
    if not eq: return jsonify({"success": False, "error": "Nessuna equazione fornita."})
    try:
        _, ordine, _, _, _ = parsifica_input(eq)
        return jsonify({"success": True, "ordine": ordine})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@equazioni_alle_differenze_bp.route('/api/equazioni_alle_differenze', methods=['POST'])
def api_eq_alle_differenze():
    data = request.get_json()
    eq = data.get("equazione", "").strip()
    if not eq: return jsonify({"success": False, "error": "Nessuna equazione fornita."})
    
    use_generic = data.get("generic_conditions", True)
    
    # Validazione preliminare per ottenere l'ordine
    try:
        _, ordine, _, _, _ = parsifica_input(eq)
    except Exception as e:
        return jsonify({"success": False, "error": f"Errore nel parsing: {e}"})

    condizioni = {}
    try:
        # Leggiamo un numero di condizioni proporzionale all'ordine
        for i in range(ordine + 5):
            val = data.get(f"y{i}", "").strip()
            if val: condizioni[f"y{i}"] = _parse_valore(val)
    except Exception as e:
        return jsonify({"success": False, "error": f"Errore condizioni: {e}"})
        
    # VALIDAZIONE RIGIDA SE NON SI USANO CONDIZIONI GENERICHE
    if not use_generic:
        num_cond = len(condizioni)
        if num_cond != ordine:
            return jsonify({
                "success": False,
                "error": f"L'equazione è di ordine {ordine}, ma hai fornito {num_cond} condizioni. "
                         f"Devi inserire esattamente {ordine} condizioni: da y(0) a y({ordine-1})."
            })
        # Verifica che siano consecutive da 0 a ordine-1
        for i in range(ordine):
            if f'y{i}' not in condizioni:
                return jsonify({
                    "success": False,
                    "error": f"Manca la condizione y({i}). Inserisci tutte le condizioni da y(0) a y({ordine-1})."
                })

    return jsonify(risolvi_eq_alle_differenze(eq, condizioni or None, use_generic))
