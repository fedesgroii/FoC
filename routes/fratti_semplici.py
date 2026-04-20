import string
import sympy as sp
from flask import Blueprint, request, jsonify

fratti_semplici_bp = Blueprint("fratti_semplici", __name__)

def to_latex(expr):
    return sp.latex(expr)

@fratti_semplici_bp.route('/api/fratti_semplici', methods=['POST'])
def api_fratti_semplici():
    data = request.get_json()
    numerator_str = data.get("numerator", "")
    denominator_str = data.get("denominator", "")

    if not numerator_str or not denominator_str:
        return jsonify({"success": False, "error": "Numeratore e denominatore sono richiesti."})

    try:
        s = sp.symbols('s')
        # Pre-process strings
        # replace ^ with ** and ensure it's treated as a symbolic expression
        n_clean = numerator_str.replace('^', '**')
        d_clean = denominator_str.replace('^', '**')
        
        N = sp.parse_expr(n_clean, local_dict={'s': s})
        D = sp.parse_expr(d_clean, local_dict={'s': s})

        if D == 0:
            return jsonify({"success": False, "error": "Il denominatore non può essere zero."})

        latex_steps = []

        # STEP 1: Verifica del grado
        deg_N = sp.degree(N, s)
        deg_D = sp.degree(D, s)
        Q = 0
        R = N
        if deg_N >= deg_D:
            Q, R = sp.div(N, D, s)
            latex_steps.append({
                "title": "Step 1: Verifica del grado e Divisione Polinomiale",
                "content": rf"\text{{Poiché grado(N) }} ({deg_N}) \ge \text{{ grado(D) }} ({deg_D}) \text{{, eseguiamo la divisione:}}" + 
                           rf" \\ F(s) = {to_latex(Q)} + \frac{{{to_latex(R)}}}{{{to_latex(D)}}}"
            })
        else:
            latex_steps.append({
                "title": "Step 1: Verifica del grado",
                "content": rf"\text{{Grado(N)}} = {deg_N} < \text{{Grado(D)}} = {deg_D}. \text{{ Nessuna divisione necessaria.}}"
            })

        # STEP 2: Fattorizzazione del denominatore
        D_factored = sp.factor(D, s)
        latex_steps.append({
            "title": "Step 2: Fattorizzazione del denominatore",
            "content": rf"D(s) = {to_latex(D_factored)}"
        })

        # STEP 3 & 4: Imposta la struttura della decomposizione
        factors = sp.factor_list(D, s)[1]
        terms = []
        constants = []
        
        # Determine number of constants needed
        # The number of constants is exactly deg_D
        num_constants = deg_D
        all_const_syms = []
        if num_constants <= 26:
            all_const_syms = [sp.Symbol(string.ascii_uppercase[i]) for i in range(num_constants)]
        else:
            all_const_syms = [sp.Symbol(f"A_{{{i+1}}}") for i in range(num_constants)]
            
        c_idx = 0
        for factor, mult in factors:
            f_deg = sp.degree(factor, s)
            if f_deg == 1:
                # Linear factor (s - p)^mult
                for i in range(1, mult + 1):
                    if c_idx < len(all_const_syms):
                        c = all_const_syms[c_idx]
                        terms.append(c / (factor**i))
                        constants.append(c)
                        c_idx += 1
            else:
                # Quadratic factor (as^2 + bs + c)^mult
                for i in range(1, mult + 1):
                    if c_idx + 1 < len(all_const_syms):
                        c1 = all_const_syms[c_idx]
                        c2 = all_const_syms[c_idx+1]
                        terms.append((c1 * s + c2) / (factor**i))
                        constants.extend([c1, c2])
                        c_idx += 2
        
        struct_expr = sp.Add(*terms, evaluate=False)
        latex_steps.append({
            "title": "Step 3 & 4: Struttura della decomposizione",
            "content": rf"\frac{{{to_latex(R)}}}{{{to_latex(D)}}} = {to_latex(struct_expr)}"
        })

        # STEP 5: Moltiplica per il denominatore comune
        # N_eq = R = struct_expr * D
        # To show it nicely, we multiply each term by D
        eq_terms = []
        for term in terms:
            eq_terms.append(sp.simplify(term * D))
        
        eq_rhs = sp.Add(*eq_terms, evaluate=False)
        latex_steps.append({
            "title": "Step 5: Moltiplicazione per il denominatore comune",
            "content": rf"{to_latex(R)} = {to_latex(eq_rhs)}"
        })

        # STEP 6: Determina i coefficienti incogniti
        # Expand RHS to compare coefficients
        eq_rhs_expanded = sp.expand(eq_rhs)
        coeffs_system = []
        for i in range(deg_D):
            eq = sp.Eq(sp.collect(R, s).coeff(s, i), sp.collect(eq_rhs_expanded, s).coeff(s, i))
            coeffs_system.append(eq)
        
        sol = sp.solve(coeffs_system, constants)
        
        # If solve returns a list, take the first one
        if isinstance(sol, list):
            sol = sol[0]
            
        sol_latex = []
        for c in constants:
            val = sol.get(c, 0)
            sol_latex.append(rf"{to_latex(c)} = {to_latex(val)}")
        
        latex_steps.append({
            "title": "Step 6: Sistema di equazioni e coefficienti",
            "content": r" \begin{cases} " + r" \\ ".join([to_latex(eq) for eq in coeffs_system]) + r" \end{cases} " + 
                       r" \implies " + r", \ ".join(sol_latex)
        })

        # STEP 7: Risultato finale
        # Use sp.apart to get a clean final version as well, but substituted struct is what's asked
        final_decomp_part = struct_expr.subs(sol)
        final_result = Q + final_decomp_part
        
        latex_steps.append({
            "title": "Step 7: Risultato finale",
            "content": rf"F(s) = {to_latex(final_result)}"
        })

        return jsonify({"success": True, "latex": latex_steps})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
