from flask import Blueprint, request, jsonify
import sympy as sp
import re
from .utils import transformations_basic as transformations
from sympy.parsing.sympy_parser import parse_expr

da_soluzione_a_sistema_bp = Blueprint("da_soluzione_a_sistema", __name__)

def to_latex(expr):
    return sp.latex(expr)

def normalize_exponential(expr_str):
    """
    Converte e^(...) in exp(...)
    Mantiene compatibilità con exp(...) già presente
    Esempi:
    - e^(2*t) → exp(2*t)
    - e^(t+1) → exp(t+1)
    - exp(t) → exp(t) (invariato)
    """
    if not isinstance(expr_str, str):
        return expr_str
    
    # e^(...) → exp(...)
    expr_str = re.sub(r'e\^\(([^)]+)\)', r'exp(\1)', expr_str)
    # e**(...) → exp(...)
    expr_str = re.sub(r'e\*\*\(([^)]+)\)', r'exp(\1)', expr_str)
    
    return expr_str

def fix_latex_y_symbols(latex_str, time_type, n, is_derivative=True):
    # Replaces y_0, y_1... with proper derivatives or time shifts
    res = latex_str
    for i in reversed(range(n + 1)):
        if time_type == 'continuous':
            if i == 0:
                res = res.replace(f"y_{{{i}}}", "y(t)")
                res = res.replace(f"y_{i}", "y(t)")
            elif i == 1:
                res = res.replace(f"y_{{{i}}}", "\\dot{y}(t)")
                res = res.replace(f"y_{i}", "\\dot{y}(t)")
            elif i == 2:
                res = res.replace(f"y_{{{i}}}", "\\ddot{y}(t)")
                res = res.replace(f"y_{i}", "\\ddot{y}(t)")
            else:
                res = res.replace(f"y_{{{i}}}", f"y^{{({i})}}(t)")
                res = res.replace(f"y_{i}", f"y^{{({i})}}(t)")
        else:
            if i == 0:
                res = res.replace(f"y_{{{i}}}", "y(t)")
                res = res.replace(f"y_{i}", "y(t)")
            else:
                res = res.replace(f"y_{{{i}}}", f"y(t+{i})")
                res = res.replace(f"y_{i}", f"y(t+{i})")
    return res

@da_soluzione_a_sistema_bp.route('/api/da_soluzione_a_sistema', methods=['POST'])
def api_da_soluzione_a_sistema():
    data = request.get_json()
    y_input = data.get("soluzione", "")
    time_type = data.get("time_type", "continuous") # continuous or discrete

    if not y_input.strip():
        return jsonify({"success": False, "error": "Nessuna soluzione fornita."})

    try:
        # Pre-process input
        y_input_clean = y_input.replace('^', '**')
        # find constants c1, c2, C1, C2, etc
        constants_str_found = list(set(re.findall(r'[cC]\d+', y_input_clean)))
        constants_str = sorted([c.lower() for c in constants_str_found], key=lambda x: int(x[1:]))
        # Ensure we only have unique lowercase constants
        constants_str = list(dict.fromkeys(constants_str))
        n = len(constants_str)

        if n == 0:
            return jsonify({"success": False, "error": "Nessuna costante c_i trovata nella soluzione. Assicurati di usare il formato c1, c2, ecc."})

        # Replace all uppercase C with lowercase c in the input string to match our symbols
        y_input_clean = re.sub(r'C(\d+)', r'c\1', y_input_clean)

        # Parse expression
        if time_type == 'continuous':
            t = sp.symbols('t', real=True)
        else:
            t = sp.symbols('t', integer=True)
            
        local_dict = {'t': t, 'e': sp.exp(1)}
        for c in constants_str:
            local_dict[c] = sp.symbols(c)

        # MODIFICA 1: Normalizza esponenziali (e^(...) → exp(...))
        y_input_normalized = normalize_exponential(y_input_clean)
        
        # MODIFICA 2: Espande automaticamente le espressioni fattorizzate
        y_expr = parse_expr(y_input_normalized, local_dict=local_dict, transformations=transformations)
        y_expr = sp.expand(y_expr)  # Espande exp(t)*(c1 + c2*t) → c1*exp(t) + c2*t*exp(t)
        
        latex_steps = []

        # 0. Sistema inserito
        time_domain_str = "Tempo Continuo (\\(\\mathbb{R}\\))" if time_type == 'continuous' else "Tempo Discreto (\\(\\mathbb{Z}\\))"
        latex_steps.append({
            "title": f"Soluzione Inserita - {time_domain_str}",
            "content": f"y(t) = {to_latex(y_expr)}"
        })

        # 1. Ordine n
        latex_steps.append({
            "title": "Ordine del sistema \\(n\\)",
            "content": f"n = {n} \\quad \\text{{(Costanti: }} {', '.join(constants_str)} \\text{{)}}"
        })

        # 2. Generazione Equazioni e Derivate/Incrementi
        equations = [y_expr]
        for i in range(1, n + 1):
            if time_type == 'continuous':
                equations.append(sp.diff(equations[-1], t))
            else:
                equations.append(y_expr.subs(t, t + i))
        
        y_syms = [sp.symbols(f"y_{i}") for i in range(n + 1)]
        
        eqs_latex = []
        for i in range(n + 1):
            eq_str = f"y_{i} = {to_latex(equations[i])}"
            eqs_latex.append(fix_latex_y_symbols(eq_str, time_type, n))
            
        latex_steps.append({
            "title": "Equazioni" if time_type == 'continuous' else "Incrementi",
            "content": "\\begin{aligned} " + " \\\\ ".join(eqs_latex) + " \\end{aligned}"
        })

        # 3. Sistema in forma matriciale
        C_syms = [local_dict[c] for c in constants_str]

        A = sp.zeros(n, n)
        b = sp.zeros(n, 1)

        for i in range(n):
            eq = equations[i]
            for j, c_sym in enumerate(C_syms):
                A[i, j] = sp.simplify(eq.coeff(c_sym))
            
            known_terms = eq
            for c_sym in C_syms:
                known_terms = known_terms.subs(c_sym, 0)
            
            b[i, 0] = sp.simplify(y_syms[i] - known_terms)

        c_syms_vec = sp.Matrix([c_sym for c_sym in C_syms])
        
        # MODIFICA 3: Fix impaginazione LaTeX con \[ \] e a capo
        step3_content = (
            r"\[ \vec{b} = A \cdot \vec{c} \] \\\\"
            r"\[ " + to_latex(b) + r" = " + to_latex(A) + r" \cdot " + to_latex(c_syms_vec) + r" \]"
        )
        latex_steps.append({
            "title": "Sistema in forma matriciale",
            "content": fix_latex_y_symbols(step3_content, time_type, n)
        })

        # 4. Calcolo del determinante
        det_A = sp.simplify(A.det())
        det_is_zero = (sp.simplify(det_A) == 0)
        
        color = "red" if det_is_zero else "green"
        det_msg = "\\det(A) = 0 \\rightarrow \\text{matrice singolare}" if det_is_zero else "\\det(A) \\neq 0 \\rightarrow \\text{matrice invertibile}"
        
        latex_steps.append({
            "title": "Calcolo del determinante",
            "content": (
                f"\\[ \\det(A) = {to_latex(det_A)} \\] \\\\"
                f"\\textcolor{{{color}}}{{{det_msg}}}"
            )
        })
        
        if det_is_zero:
            return jsonify({"success": False, "error": "Impossibile isolare le costanti. Il determinante della matrice A è zero (costanti linearmente dipendenti)."})

        # 5. Matrice inversa
        cofactor_matrix = A.cofactor_matrix()
        adj_A = sp.simplify(cofactor_matrix.T)
        A_inv = sp.simplify(adj_A / det_A)

        # MODIFICA 3 (continua): Fix impaginazione con \[ \] e a capo tra elementi
        latex_steps.append({
            "title": "Matrice inversa",
            "content": (
                f"\\[ \\text{{Matrice dei cofattori: }} \\text{{Cof}}(A) = {to_latex(cofactor_matrix)} \\] \\\\"
                f"\\[ \\text{{Matrice aggiunta: }} \\text{{Adj}}(A) = \\text{{Cof}}(A)^T = {to_latex(adj_A)} \\] \\\\"
                f"\\[ A^{{-1}} = \\frac{{1}}{{\\det(A)}} \\cdot \\text{{Adj}}(A) = {to_latex(A_inv)} \\]"
            )
        })

        # 6. Calcolo delle costanti
        c_vector = sp.simplify(A_inv * b)
        
        sol_dict = {}
        for i, c_sym in enumerate(C_syms):
            sol_dict[c_sym] = sp.simplify(c_vector[i])

        # MODIFICA 3 (continua): Fix impaginazione
        step6_content = (
            r"\[ \vec{c} = A^{-1} \cdot \vec{b} \] \\"
            r"\[ " + to_latex(c_syms_vec) + r" = " + to_latex(A_inv) + r" \cdot " + to_latex(b) + r" = " + to_latex(c_vector) + r" \]"
        )
        latex_steps.append({
            "title": "Calcolo delle costanti",
            "content": fix_latex_y_symbols(step6_content, time_type, n)
        })

        # 7. Costanti isolate
        sol_latex = []
        for c_sym in C_syms:
            if c_sym in sol_dict:
                sol_str = f"{to_latex(c_sym)} = {to_latex(sol_dict[c_sym])}"
                sol_latex.append(fix_latex_y_symbols(sol_str, time_type, n))
        
        latex_steps.append({
            "title": "Costanti isolate",
            "content": "\\begin{aligned} " + " \\\\ ".join(sol_latex) + " \\end{aligned}"
        })

        # 4. Equazione finale
        y_n_expr = equations[n].subs(sol_dict)
        y_n_expr = sp.simplify(sp.expand(y_n_expr))
        
        # Build the final equation y_n = ...
        final_eq_str = f"y_{n} = {to_latex(y_n_expr)}"
        latex_steps.append({
            "title": "Equazione differenziale finale" if time_type == 'continuous' else "Equazione alle differenze finale",
            "content": fix_latex_y_symbols(final_eq_str, time_type, n)
        })

        # 5. Variabili di Stato
        state_vars = []
        for i in range(n):
            if time_type == 'continuous':
                if i == 0:
                    state_vars.append(f"x_{{{i+1}}}(t) = y(t)")
                elif i == 1:
                    state_vars.append(f"x_{{{i+1}}}(t) = \\dot{{y}}(t)")
                elif i == 2:
                    state_vars.append(f"x_{{{i+1}}}(t) = \\ddot{{y}}(t)")
                else:
                    state_vars.append(f"x_{{{i+1}}}(t) = y^{{({i})}}(t)")
            else:
                if i == 0:
                    state_vars.append(f"x_{{{i+1}}}(k) = y(k)")
                else:
                    state_vars.append(f"x_{{{i+1}}}(k) = y(k+{i})")
        
        latex_steps.append({
            "title": "Variabili di stato introdotte",
            "content": "\\begin{aligned} " + " \\\\ ".join(state_vars) + " \\end{aligned}"
        })

        # 6. Matrici Finali
        coeffs = [sp.simplify(y_n_expr.diff(y_syms[i])) for i in range(n)]
        u_term = sp.simplify(y_n_expr - sum(coeffs[i]*y_syms[i] for i in range(n)))
        
        A = sp.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1
        for i in range(n):
            A[n - 1, i] = coeffs[i]
            
        B = sp.zeros(n, 1)
        if u_term != 0:
            B[n - 1, 0] = 1 # Assuming input u(t) acts on the last state variable
        
        C = sp.zeros(1, n)
        C[0, 0] = 1
        
        D = sp.zeros(1, 1) # D is 0

        html_content = f"""
        <div style="display: flex; flex-direction: column; gap: 20px; align-items: center; margin-top: 15px;">
            <div style="display: flex; gap: 50px; justify-content: center; align-items: center;">
                <div>\\[ A = {to_latex(A)} \\]</div>
                <div>\\[ B = {to_latex(B)} \\]</div>
            </div>
            <div style="display: flex; gap: 50px; justify-content: center; align-items: center;">
                <div>\\[ C = {to_latex(C)} \\]</div>
                <div>\\[ D = {to_latex(D)} \\]</div>
            </div>
        </div>
        """
        
        if u_term != 0:
            html_content += f"<br><br><div style='text-align:center; font-size: 18px;'>Con ingresso \\( u(t) = {to_latex(u_term)} \\)</div>"
        else:
            html_content += f"<br><br><div style='text-align:center; font-size: 18px;'>Sistema autonomo (ingresso \\( u(t) = 0 \\))</div>"

        latex_steps.append({
            "title": "Matrici del sistema in forma compagna",
            "content": html_content,
            "is_html": True
        })

        return jsonify({"success": True, "latex": latex_steps})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})