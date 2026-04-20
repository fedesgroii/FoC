import sympy as sp
from sympy.abc import s
from flask import Blueprint, request, jsonify
import re

fratti_semplici_bp = Blueprint("fratti_semplici", __name__)

def to_latex(expr):
    """Converte un'espressione sympy in stringa LaTeX."""
    return sp.latex(expr)

def parse_polynomial(expr_str):
    """Pulisce e parsa una stringa in un'espressione sympy."""
    # Sostituisce ^ con **
    clean = expr_str.replace('^', '**')
    # Gestisce moltiplicazioni implicite (es: 3s -> 3*s)
    clean = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', clean)
    # Rimuove spazi superflui
    clean = clean.strip()
    return sp.parse_expr(clean, local_dict={'s': s})

@fratti_semplici_bp.route('/api/fratti_semplici', methods=['POST'])
def api_fratti_semplici():
    data = request.get_json()
    num_str = data.get("numerator", "")
    den_str = data.get("denominator", "")

    if not num_str or not den_str:
        return jsonify({"success": False, "error": "Numeratore e denominatore sono richiesti."})

    try:
        # STEP 0: Parsing e Verifica Preliminare
        N = parse_polynomial(num_str)
        D = parse_polynomial(den_str)

        if D == 0:
            return jsonify({"success": False, "error": "Il denominatore non può essere zero."})
        latex_steps = []
        
        # PASSO 0: FUNZIONE INSERITA
        latex_steps.append({
            "title": "📝 FUNZIONE INSERITA",
            "content": rf"F(s) = \frac{{{to_latex(N)}}}{{{to_latex(D)}}}",
            "type": "initial"
        })
        
        deg_N = sp.degree(N, s)
        deg_D = sp.degree(D, s)
        
        Q, R = 0, N
        if deg_N >= deg_D:
            Q, R = sp.div(N, D, s)
            latex_steps.append({
                "title": "Verifica Preliminare (Divisione)",
                "content": rf"\text{{Grado(N)}} \ge \text{{Grado(D)}} \implies F(s) = {to_latex(Q)} + \frac{{{to_latex(R)}}}{{{to_latex(D)}}}"
            })
        else:
            latex_steps.append({
                "title": "Verifica Preliminare",
                "content": rf"\deg(N) < \deg(D) \implies \text{{Funzione già propria.}}"
            })

        # STEP 1: Scomposizione del Denominatore
        D_parts = sp.factor_list(D)
        factors_data = D_parts[1]
        D_factored = sp.factor(D)
        
        classification = []
        for factor, mult in factors_data:
            f_deg = sp.degree(factor, s)
            if f_deg == 1:
                p = sp.solve(factor, s)[0]
                if mult == 1:
                    classification.append(rf"\text{{Polo semplice: }} s = {to_latex(p)}")
                else:
                    classification.append(rf"\text{{Polo multiplo (x{mult}): }} s = {to_latex(p)}")
            elif f_deg == 2:
                classification.append(rf"\text{{Poli complessi coniugati: }} {to_latex(factor)}")

        latex_steps.append({
            "title": "Scomposizione Denominatore",
            "content": rf"D(s) = {to_latex(D_factored)} \\ \text{{Analisi: }} " + r", \ ".join(classification)
        })

        # STEP 2: Scrittura Forma Generica
        generic_terms = []
        constants = []
        const_idx = 1
        
        for factor, mult in factors_data:
            f_deg = sp.degree(factor, s)
            if f_deg == 1:
                for i in range(1, mult + 1):
                    c = sp.Symbol(f"A_{const_idx}")
                    generic_terms.append(c / (factor**i))
                    constants.append(c)
                    const_idx += 1
            elif f_deg == 2:
                for i in range(1, mult + 1):
                    b = sp.Symbol(f"B_{const_idx}")
                    c = sp.Symbol(f"C_{const_idx}")
                    generic_terms.append((b * s + c) / (factor**i))
                    constants.extend([b, c])
                    const_idx += 1
        
        struct_expr = sp.Add(*generic_terms, evaluate=False)
        latex_steps.append({
            "title": "Forma della Decomposizione",
            "content": rf"\frac{{{to_latex(R)}}}{{{to_latex(D)}}} = {to_latex(struct_expr)}"
        })

        # STEP 3: Calcolo Coefficienti
        eq_rhs_sum = 0
        for term in generic_terms:
            eq_rhs_sum += sp.simplify(term * D)
        
        eq_rhs = sp.expand(eq_rhs_sum)
        coeffs_system = []
        for i in range(deg_D):
            lhs_c = sp.collect(R, s).coeff(s, i)
            rhs_c = sp.collect(eq_rhs, s).coeff(s, i)
            coeffs_system.append(sp.Eq(lhs_c, rhs_c))
            
        sol = sp.solve(coeffs_system, constants)
        if isinstance(sol, list):
            sol = sol[0] if sol else {}

        sol_items = [rf"{to_latex(c)} = {to_latex(sol.get(c, 0))}" for c in constants]
        
        latex_steps.append({
            "title": "Calcolo dei Coefficienti",
            "content": rf"\begin{{cases}} " + 
                       r" \\ ".join([to_latex(eq) for eq in coeffs_system]) + 
                       r" \end{{cases}} \implies " + r", \ ".join(sol_items)
        })

        # STEP 4: Assembly del Risultato
        final_decomp = struct_expr.subs(sol)
        final_result = Q + final_decomp
        
        # RISULTATO FINALE
        latex_steps.append({
            "title": "✅ RISULTATO FINALE",
            "content": rf"F(s) = {to_latex(final_result)}",
            "type": "final"
        })

        return jsonify({"success": True, "latex": latex_steps})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
