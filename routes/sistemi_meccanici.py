from flask import Blueprint, request, jsonify
import sympy as sp
import re
from sympy import symbols, diff, Matrix, solve, sympify
from sympy.parsing.sympy_parser import parse_expr
from .utils import transformations, sostituisci_pedici, parse_frazioni_complete

sistemi_meccanici_bp = Blueprint("sistemi_meccanici", __name__)

@sistemi_meccanici_bp.route('/api/sistemi-meccanici', methods=['POST'])
def calcola_spazio_stato():
    try:
        data = request.get_json()
        equazioni_raw = data.get('equazioni', [])
        
        # Pulizia equazioni
        equazioni_pulite = []
        for eq in equazioni_raw:
            if not eq.strip():
                continue
            # Sostituisci \dot q1 -> dot_q1, \ddot q1 -> ddot_q1
            # Rimuove \ prima di dot e ddot
            eq = eq.replace('\\dot ', 'dot_').replace('\\ddot ', 'ddot_')
            eq = eq.replace('\\dot', 'dot_').replace('\\ddot', 'ddot_')
            # Rimuove eventuali spazi tra dot_ e la variabile
            eq = re.sub(r'dot_ +', 'dot_', eq)
            eq = re.sub(r'ddot_ +', 'ddot_', eq)
            equazioni_pulite.append(eq)

        if not equazioni_pulite:
            return jsonify({"success": False, "errore": "Nessuna equazione inserita."})

        # Identificazione gradi di libertà (q1, q2, ...)
        # Cerchiamo pattern q[numero] o dot_q[numero] o ddot_q[numero]
        q_vars = set()
        for eq in equazioni_pulite:
            matches = re.findall(r'(?:ddot_|dot_)?q(\d+)', eq)
            for m in matches:
                q_vars.add(int(m))
        
        if not q_vars:
            return jsonify({"success": False, "errore": "Impossibile identificare le variabili q1, q2, ..."})

        num_dof = max(q_vars)
        n_states = 2 * num_dof

        # Definizione simboli
        q = [symbols(f'q{i+1}') for i in range(num_dof)]
        dot_q = [symbols(f'dot_q{i+1}') for i in range(num_dof)]
        ddot_q = [symbols(f'ddot_q{i+1}') for i in range(num_dof)]
        u = symbols('u')
        
        # Parametri (m, k, d) - li definiamo come simboli generici se compaiono
        # Oppure li definiamo dinamicamente durante il parsing
        
        # Parsing delle equazioni
        sympy_eqs = []
        for eq_str in equazioni_pulite:
            # Gestione '='
            if '=' in eq_str:
                lhs_str, rhs_str = eq_str.split('=')
                # Trasforma in expr = lhs - rhs = 0
                eq_str = f"({lhs_str}) - ({rhs_str})"
            
            # Pre-processing stile FoC
            eq_str = sostituisci_pedici(eq_str)
            eq_str = parse_frazioni_complete(eq_str)
            
            # Creazione dizionario locale per parse_expr
            local_dict = {f'q{i+1}': q[i] for i in range(num_dof)}
            local_dict.update({f'dot_q{i+1}': dot_q[i] for i in range(num_dof)})
            local_dict.update({f'ddot_q{i+1}': ddot_q[i] for i in range(num_dof)})
            local_dict['u'] = u
            
            # Identifica altri simboli (m1, k1, etc) e aggiungili come simboli
            # Qualunque parola che non sia q, dot_q, ddot_q, u, sin, cos, exp, log, sqrt
            words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', eq_str)
            for w in words:
                if w not in local_dict and w not in ['sin', 'cos', 'exp', 'log', 'sqrt', 'tan']:
                    local_dict[w] = symbols(w)

            expr = parse_expr(eq_str, transformations=transformations, local_dict=local_dict)
            sympy_eqs.append(expr)

        # Risoluzione per ddot_q
        # Dobbiamo avere tante equazioni quanti dof (idealmente)
        try:
            sol_ddot = solve(sympy_eqs, ddot_q)
            if not sol_ddot:
                # Prova a risolvere singolarmente se solve() globale fallisce
                sol_ddot = {}
                for i in range(num_dof):
                    for eq in sympy_eqs:
                        if ddot_q[i] in eq.free_symbols:
                            s = solve(eq, ddot_q[i])
                            if s:
                                sol_ddot[ddot_q[i]] = s[0]
                                break
        except Exception as e:
            return jsonify({"success": False, "errore": f"Errore nella risoluzione delle equazioni: {str(e)}"})

        if not sol_ddot:
            return jsonify({"success": False, "errore": "Impossibile isolare i termini in ddot(q)."})

        # Definizione variabili di stato x1, x2, ...
        # x_odd = q, x_even = dot_q
        x = symbols(f'x1:{n_states+1}')
        
        # Mapping: q_i -> x_{2i-1}, dot_q_i -> x_{2i}
        state_mapping = {}
        for i in range(num_dof):
            state_mapping[q[i]] = x[2*i]
            state_mapping[dot_q[i]] = x[2*i+1]

        # Costruzione equazioni di stato dot_x = Ax + Bu
        # dot_x1 = x2
        # dot_x2 = espressione di ddot_q1 (sostituendo q, dot_q con x)
        # dot_x3 = x4
        # dot_x4 = espressione di ddot_q2 ...
        
        dot_x_exprs = []
        for i in range(num_dof):
            # dot_x_{2i+1} = x_{2i+2}
            dot_x_exprs.append(x[2*i+1])
            
            # dot_x_{2i+2} = ddot_q_{i+1}
            if isinstance(sol_ddot, dict):
                expr_ddot = sol_ddot.get(ddot_q[i], 0)
            elif isinstance(sol_ddot, list) and len(sol_ddot) > 0:
                expr_ddot = sol_ddot[0].get(ddot_q[i], 0)
            else:
                expr_ddot = 0
            
            # Sostituzione variabili di stato
            expr_ddot_x = expr_ddot.subs(state_mapping)
            dot_x_exprs.append(expr_ddot_x)

        # Costruizione Matrici A e B
        A = Matrix([[diff(expr, xi) for xi in x] for expr in dot_x_exprs])
        B = Matrix([[diff(expr, u)] for expr in dot_x_exprs])

        # Semplificazione e conversione in LaTeX
        A = A.applyfunc(sp.simplify)
        B = B.applyfunc(sp.simplify)

        def to_latex_matrix(M):
            # Use nsimplify to keep fractions
            M_simplified = M.applyfunc(lambda x: sp.nsimplify(x, rational=True))
            rows = []
            for i in range(M_simplified.rows):
                row_items = [sp.latex(M_simplified[i, j]) for j in range(M_simplified.cols)]
                rows.append(" & ".join(row_items))
            return "\\begin{pmatrix} " + " \\\\ ".join(rows) + " \\end{pmatrix}"

        # Formattazione per le equazioni di stato (sostituisci x_i con dot_x_i)
        def format_dot_x(i, expr):
            expr_latex = sp.latex(sp.nsimplify(expr, rational=True))
            # Sostituzione estetica x1, x2 -> x_{1}, x_{2}
            expr_latex = re.sub(r'x(\d+)', r'x_{\1}', expr_latex)
            return f"\\dot{{x}}_{{{i+1}}} = {expr_latex}"

        res_latex = [
            {
                "title": "Definizione delle variabili di stato:",
                "content": r"\mathbf{x} = \begin{bmatrix} " + " \\\\ ".join([f"x_{{{i+1}}}" for i in range(n_states)]) + r" \end{bmatrix} = \begin{bmatrix} " + " \\\\ ".join([f"q_{{{i//2+1}}}" if i%2==0 else f"\\dot{{q}}_{{{i//2+1}}}" for i in range(n_states)]) + r" \end{bmatrix}"
            },
            {
                "title": "Equazioni di stato isolate:",
                "content": r"\begin{cases} " + " \\\\ ".join([format_dot_x(i, expr) for i, expr in enumerate(dot_x_exprs)]) + r" \end{cases}"
            },
            {
                "title": "Rappresentazione in spazio di stato (A, B):",
                "content": f"A = {to_latex_matrix(A)}, \\quad B = {to_latex_matrix(B)}"
            }
        ]

        return jsonify({
            "success": True,
            "latex": res_latex
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "errore": str(e)})
