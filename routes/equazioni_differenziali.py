from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import (
    symbols, Function, Eq, dsolve, Derivative, exp, simplify, collect,
    cos, sin, latex, expand, roots, solve, I
)
from sympy.parsing.sympy_parser import parse_expr
from .utils import transformations_basic as transformations
import re

equazioni_differenziali_bp = Blueprint("equazioni_differenziali", __name__)


@equazioni_differenziali_bp.route('/api/equazioni_differenziali', methods=['POST'])
def api_equazione_differenziale():
    data = request.get_json()
    equation = data.get("equazione", "")
    condizioni = data.get("condizioniIniziali", [])
    applica_condizioni = data.get("applicaCondizioni", False)

    if not equation.strip():
        return jsonify({"success": False, "error": "Nessuna equazione differenziale fornita."})

    result = solve_differential_equation(equation, condizioni if applica_condizioni else None)
    return jsonify(result)

def solve_differential_equation(equation_str, conditions=None):
    """
    Risolve un'equazione differenziale lineare a coefficienti costanti
    utilizzando il metodo dell'operatore Delta.
    """
    try:
        import sympy as sp
        from sympy import (
            symbols, Function, Eq, dsolve, Derivative, exp, simplify, collect,
            cos, sin, latex, expand, roots, solve, I
        )
        from sympy.parsing.sympy_parser import parse_expr
        
        t = symbols('t', real=True)
        y = Function('y')(t)
        d = symbols('d') # Rappresenta l'operatore Δ
        
        # 1. Parsing dell'input
        if '=' not in equation_str:
            raise ValueError("L'equazione deve contenere un segno '='")
        
        lhs_str, rhs_str = equation_str.split('=', 1)
        
        # Pulizia stringhe LHS
        lhs_str = lhs_str.replace('\\', '')
        # Sostituzioni per notazione y', y'', y''' e y^{(n)}
        lhs_str = re.sub(r"y\^\{\((\d+)\)\}", lambda m: f"(d**{m.group(1)})", lhs_str)
        lhs_str = lhs_str.replace("y'''", "(d**3)")
        lhs_str = lhs_str.replace("y''", "(d**2)")
        lhs_str = lhs_str.replace("y'", "d")
        lhs_str = lhs_str.replace('Δ', 'd').replace('y(t)', '1').replace('y', '1')
        
        # Rimuove moltiplicazioni rimaste appese e pulisce
        lhs_str = lhs_str.strip().rstrip('*').strip()
        if not lhs_str: lhs_str = "d" # Default se vuoto (dy/dt)
        
        # Pulizia stringhe RHS
        # Conversione funzioni LaTeX
        rhs_str = rhs_str.replace(r'\sin', 'sin').replace(r'\cos', 'cos').replace(r'\tan', 'tan').replace(r'\exp', 'exp')
        # Sostituzioni frazioni
        rhs_str = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', rhs_str)
        
        # Gestione potenze: t^{2} -> t**(2)
        rhs_str = re.sub(r'\^\{([^}]+)\}', r'**(\1)', rhs_str)
        rhs_str = rhs_str.replace('^', '**').replace('e**', 'exp')
        
        # Aggiunta spazi impliciti per moltiplicazione (gestita meglio dal sympy parser)
        rhs_str = re.sub(r'e\*\*\((.*?)\)', r'exp(\1)', rhs_str)
        
        # Parsing dei polinomi
        from sympy.parsing.sympy_parser import implicit_multiplication_application
        my_transformations = transformations + (implicit_multiplication_application,)
        
        local_dict = {'d': d, 't': t, 'exp': exp, 'cos': cos, 'sin': sin, 'I': I, 'e': exp(1)}
        try:
            P_d = parse_expr(lhs_str, local_dict=local_dict, transformations=my_transformations)
            u_t = parse_expr(rhs_str, local_dict=local_dict, transformations=my_transformations)
        except Exception as e:
            raise ValueError(f"Errore nel parsing dell'equazione: {str(e)}")
        
        P_d = expand(P_d)
        ordine = sp.degree(P_d, d)
        
        latex_steps = []
        
        # Step 1: Identificazione
        latex_steps.append({
            "title": "Identificazione dell'equazione:",
            "content": rf"\text{{Equazione lineare a coefficienti costanti di ordine }} {ordine}: P(\Delta)y(t) = u(t)"
        })
        
        # Step 2: Soluzione Omogenea
        # Polinomio caratteristico
        r = symbols('r')
        P_r = P_d.subs(d, r)
        latex_steps.append({
            "title": "Equazione caratteristica:",
            "content": rf"P(r) = {latex(P_r)} = 0"
        })
        
        # Radici
        r_roots = roots(P_r, r)
        # Se roots() non trova tutto (es. polinomi alto grado), usiamo solve
        if sp.simplify(sum(r_roots.values()) - ordine) != 0:
            soluzioni = solve(P_r, r)
            r_roots = {}
            for s in soluzioni:
                r_roots[s] = r_roots.get(s, 0) + 1
                
        roots_latex = ", ".join([rf"r_{i+1} = {latex(root)} \text{{ (molt. {m})}}" for i, (root, m) in enumerate(r_roots.items())])
        latex_steps.append({
            "title": "Radici del polinomio caratteristico:",
            "content": roots_latex
        })
        
        # Costruzione y_h
        y_h = 0
        c_idx = 1
        for root, m in r_roots.items():
            for i in range(m):
                c = symbols(f'c_{c_idx}')
                term = c * (t**i)
                if root.is_real:
                    term *= exp(root * t)
                else:
                    # Gestione radici complesse a + ib
                    alpha = sp.re(root)
                    beta = sp.im(root)
                    beta_numeric = sp.N(beta)
                    if beta_numeric.is_positive: # Prendi solo la parte positiva per evitare duplicati
                        c2 = symbols(f'c_{c_idx+1}')
                        term = (c * cos(beta * t) + c2 * sin(beta * t)) * (t**i) * exp(alpha * t)
                        c_idx += 1
                    else:
                        continue
                y_h += term
                c_idx += 1
        
        latex_steps.append({
            "title": "Soluzione dell'equazione omogenea \( y_o(t) \):",
            "content": rf"y_o(t) = {latex(y_h)}"
        })
        
        # Step 3: Soluzione Particolare
        y_p = 0
        if u_t.is_zero or sp.simplify(u_t) == 0:
            latex_steps.append({
                "title": "Soluzione particolare \( y_p(t) \):",
                "content": r"y_p(t) = 0 \quad \text{(Equazione omogenea)}"
            })
        else:
            # Usiamo dsolve di SymPy per trovare la soluzione completa e poi sottraiamo l'omogenea
            # Questo è più robusto che implementare tutti i casi a mano
            # Convertiamo P(d) in forma differenziale per dsolve
            diff_eq_lhs = 0
            for term in P_d.as_ordered_terms():
                coeff, monom = term.as_coeff_Mul()
                deg = sp.degree(monom, d)
                if deg == 0:
                    diff_eq_lhs += coeff * y
                else:
                    diff_eq_lhs += coeff * Derivative(y, (t, deg))
            
            full_sol = dsolve(Eq(diff_eq_lhs, u_t), y)
            # Sottraiamo la parte omogenea (i termini con C1, C2...)
            y_full_expr = full_sol.rhs
            # Per estrarre y_p, mettiamo a zero tutte le costanti arbitrarie C1, C2...
            free_syms = y_full_expr.free_symbols
            subs_dict = {s: 0 for s in free_syms if s.name.startswith('C')}
            y_p = simplify(y_full_expr.subs(subs_dict))
            
            # Descrizione del metodo (semplificata)
            latex_steps.append({
                "title": "Soluzione particolare \( y_p(t) \):",
                "content": rf"y_p(t) = \frac{{1}}{{P(\Delta)}} u(t) = {latex(y_p)}"
            })
            
        # Step 4: Soluzione Generale
        y_gen = y_h + y_p
        latex_steps.append({
            "title": "Soluzione generale \( y(t) = y_o(t) + y_p(t) \):",
            "content": rf"y(t) = {latex(y_gen)}"
        })
        
        # Step 5: Condizioni Iniziali
        if conditions and len(conditions) > 0:
            cond_eqs = []
            constants = [s for s in y_gen.free_symbols if s.name.startswith('c_')]
            constants = sorted(constants, key=lambda s: int(s.name.split('_')[1]))
            
            # Se ci sono meno condizioni dell'ordine, risolviamo solo per quelle fornite
            n_cond = min(len(conditions), ordine)
            for i in range(n_cond):
                try:
                    val = parse_expr(str(conditions[i]), local_dict={'e': exp(1)})
                    # Derivata i-esima valutata in t=0
                    expr_at_0 = y_gen.diff(t, i).subs(t, 0)
                    cond_eqs.append(Eq(expr_at_0, val))
                except:
                    continue
            
            if cond_eqs:
                # Risoluzione del sistema per le costanti
                sol_const = solve(cond_eqs, constants)
                if sol_const:
                    y_particolare = y_gen.subs(sol_const)
                    
                    # Se rimangono costanti (perché fornite meno condizioni dell'ordine)
                    # le lasciamo come simboli
                    
                    latex_steps.append({
                        "title": "Applicazione delle condizioni iniziali:",
                        "content": rf"\begin{{cases}} " + r" \\ ".join([latex(eq) for eq in cond_eqs]) + r" \end{{cases}}"
                    })
                    
                    latex_steps.append({
                        "title": "Soluzione particolare finale:",
                        "content": rf"\mathbf{{y(t) = {latex(simplify(y_particolare))}}}"
                    })
        
        return {"success": True, "latex": latex_steps}

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
