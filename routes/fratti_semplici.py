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
        
        deg_N = sp.degree(N, s)
        deg_D = sp.degree(D, s)
        
        Q, R = 0, N
        if deg_N >= deg_D:
            Q, R = sp.div(N, D, s)
            latex_steps.append({
                "title": "Parsing e Verifica Preliminare",
                "content": rf"\text{{Poiché }} \deg(N) \ge \deg(D) \text{{ ({deg_N} \ge {deg_D}), eseguiamo la divisione:}} \\" +
                           rf"\frac{{N(s)}}{{D(s)}} = {to_latex(Q)} + \frac{{{to_latex(R)}}}{{{to_latex(D)}}}"
            })
        else:
            latex_steps.append({
                "title": "Parsing e Verifica Preliminare",
                "content": rf"\deg(N) = {deg_N} < \deg(D) = {deg_D}. \text{{ La funzione è già propria (non serve divisione).}}"
            })

        # STEP 1: Scomposizione del Denominatore
        # factor_list restituisce (coeff_principale, [(fattore, molteplicità), ...])
        D_parts = sp.factor_list(D)
        lead_coeff = D_parts[0]
        factors_data = D_parts[1]
        D_factored = sp.factor(D)
        
        classification = []
        for factor, mult in factors_data:
            f_deg = sp.degree(factor, s)
            if f_deg == 1:
                if mult == 1:
                    classification.append(rf"\text{{Polo reale semplice: }} s = {to_latex(sp.solve(factor, s)[0])}")
                else:
                    classification.append(rf"\text{{Polo reale multiplo: }} s = {to_latex(sp.solve(factor, s)[0])} \text{{ con molteplicità }} {mult}")
            elif f_deg == 2:
                classification.append(rf"\text{{Poli complessi coniugati (fattore irriducibile): }} {to_latex(factor)}")
            else:
                classification.append(rf"\text{{Fattore di grado {f_deg}: }} {to_latex(factor)}")

        latex_steps.append({
            "title": "Scomposizione del Denominatore",
            "content": rf"D(s) = {to_latex(D_factored)} \\ " + r" \\ ".join(classification)
        })

        # STEP 2: Scrittura Forma Generica della Scomposizione
        generic_terms = []
        constants = []
        const_idx = 1
        
        # Gestione del coefficiente principale se diverso da 1
        # Lo incorporiamo nel denominatore dei fratti per semplicità
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
        
        # Se c'è un lead_coeff, la scomposizione R/D = (1/lead_coeff) * generic_expr
        # Ma è più facile considerare R / (D/lead_coeff) = generic_expr
        # Quindi R = lead_coeff * generic_expr * (D/lead_coeff) = generic_expr * D
        struct_expr = sp.Add(*generic_terms, evaluate=False)
        latex_steps.append({
            "title": "Scrittura Forma Generica",
            "content": rf"\frac{{{to_latex(R)}}}{{{to_latex(D)}}} = {to_latex(struct_expr)}"
        })

        # STEP 3: Calcolo Coefficienti (Sistema Lineare)
        # R(s) = generic_expr * D(s)
        # Semplifichiamo ogni termine individualmente per garantire che s a denominatore si cancellino
        eq_rhs_sum = 0
        for term in generic_terms:
            eq_rhs_sum += sp.simplify(term * D)
        
        eq_rhs = sp.expand(eq_rhs_sum)
        
        # Generiamo il sistema uguagliando i coefficienti delle potenze di s
        coeffs_system = []
        # Consideriamo tutte le potenze fino a deg_D - 1
        for i in range(deg_D):
            lhs_c = sp.collect(R, s).coeff(s, i)
            rhs_c = sp.collect(eq_rhs, s).coeff(s, i)
            coeffs_system.append(sp.Eq(lhs_c, rhs_c))
            
        sol = sp.solve(coeffs_system, constants)
        
        # Se sol è una lista (rari casi con solve), prendiamo il primo dizionario
        if isinstance(sol, list):
            sol = sol[0] if sol else {}

        sol_items = [rf"{to_latex(c)} = {to_latex(sol.get(c, 0))}" for c in constants]
        
        latex_steps.append({
            "title": "Calcolo Coefficienti",
            "content": rf"\text{{Uguagliando i coefficienti:}} \\ \begin{{cases}} " + 
                       r" \\ ".join([to_latex(eq) for eq in coeffs_system]) + 
                       r" \end{cases} \\ \implies " + r", \ ".join(sol_items)
        })

        # STEP 4: Assembly del Risultato
        final_decomp = struct_expr.subs(sol)
        final_result = Q + final_decomp
        
        latex_steps.append({
            "title": "Assembly del Risultato",
            "content": rf"F(s) = {to_latex(final_result)}"
        })

        # STEP 5: Verifica
        # Ricostruiamo la frazione originale per verifica
        reconstructed = sp.together(final_decomp)
        # Nota: reconstructed avrà denominatore fattorizzato, confrontiamo con R/D
        diff = sp.simplify(R/D - final_decomp)
        is_correct = (diff == 0)
        
        verification_msg = r"\text{Verifica completata con successo! } \checkmark" if is_correct else r"\text{Attenzione: discrepanza rilevata nella verifica. } \times"
        
        latex_steps.append({
            "title": "Verifica Finale",
            "content": rf"\text{{Ricostruendo: }} \frac{{{to_latex(sp.numer(reconstructed))}}}{{{to_latex(sp.denom(reconstructed))}}} \\" +
                       rf"\text{{Differenza: }} {to_latex(diff)} \\ {verification_msg}"
        })

        return jsonify({"success": True, "latex": latex_steps})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
