import sympy as sp
from flask import Blueprint, request, jsonify
import re
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

fratti_semplici_bp = Blueprint("fratti_semplici", __name__)

def clean_input(s):
    """Pulisce l'input per sympy."""
    s = s.replace('^', '**')
    # Sostituisce eventuali virgole con punti (per decimali, sebbene insoliti in fratti semplici)
    s = s.replace(',', '.')
    return s.strip()

def get_variable(expr_n, expr_d):
    """Trova la variabile utilizzata nei polinomi, default 'x'."""
    vars_n = expr_n.atoms(sp.Symbol)
    vars_d = expr_d.atoms(sp.Symbol)
    all_vars = vars_n.union(vars_d)
    if all_vars:
        # Prendi la prima variabile alfabetica (solitamente x o s)
        return sorted(list(all_vars), key=lambda x: x.name)[0]
    return sp.Symbol('x')

def to_latex(expr):
    return sp.latex(expr)

@fratti_semplici_bp.route('/api/fratti_semplici', methods=['POST'])
def api_fratti_semplici():
    data = request.get_json()
    num_str = data.get("numerator", "")
    den_str = data.get("denominator", "")

    if not num_str or not den_str:
        return jsonify({"success": False, "error": "Numeratore e denominatore sono richiesti."})

    try:
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        # Parsing dei polinomi
        N_expr = parse_expr(clean_input(num_str), transformations=transformations)
        D_expr = parse_expr(clean_input(den_str), transformations=transformations)
        
        var = get_variable(N_expr, D_expr)
        
        if D_expr == 0:
            return jsonify({"success": False, "error": "Il denominatore non può essere zero."})

        latex_steps = []

        # PASSO 0: Input ricevuto
        latex_steps.append({
            "title": "Input ricevuto",
            "content": rf"\frac{{{to_latex(N_expr)}}}{{{to_latex(D_expr)}}}",
            "type": "initial"
        })

        # PASSO 1: Verifica proprietà
        deg_N = sp.degree(N_expr, var)
        deg_D = sp.degree(D_expr, var)
        
        Q, R = 0, N_expr
        is_improper = False
        if deg_N >= deg_D:
            is_improper = True
            Q, R = sp.div(N_expr, D_expr, var)
            latex_steps.append({
                "title": "Verifica proprietà",
                "content": rf"\text{{Grado(N)}} = {deg_N} \ge \text{{Grado(D)}} = {deg_D} \implies \text{{Divisione polinomiale:}} \\" +
                           rf"\frac{{{to_latex(N_expr)}}}{{{to_latex(D_expr)}}} = {to_latex(Q)} + \frac{{{to_latex(R)}}}{{{to_latex(D_expr)}}}"
            })
        else:
            latex_steps.append({
                "title": "Verifica proprietà",
                "content": rf"\text{{Grado(N)}} = {deg_N} < \text{{Grado(D)}} = {deg_D} \implies \text{{Frazione propria.}}"
            })

        # Se il numeratore è diventato 0 dopo la divisione, abbiamo finito
        if R == 0:
            latex_steps.append({
                "title": "Risultato finale",
                "content": rf"\frac{{{to_latex(N_expr)}}}{{{to_latex(D_expr)}}} = {to_latex(Q)}",
                "type": "final"
            })
            return jsonify({"success": True, "latex": latex_steps})

        # PASSO 2: Fattorizzazione del denominatore
        # Usiamo factor con extension=True per supportare radici reali se necessario, 
        # ma di default factor(D) over Q è lo standard per i fratti semplici scolastici.
        D_factored = sp.factor(D_expr)
        latex_steps.append({
            "title": "Fattorizzazione del denominatore",
            "content": rf"{to_latex(D_expr)} = {to_latex(D_factored)}"
        })

        # PASSO 3: Forma generale della decomposizione
        # Estraiamo i fattori per costruire la forma
        coeff_lead, factors_list = sp.factor_list(D_expr)
        # Se c'è un coefficiente direttivo diverso da 1, lo incorporiamo in D
        # In realtà per la decomposizione sympy.apart gestisce tutto, ma noi dobbiamo costruirla a mano.
        
        partial_terms = []
        unknowns = []
        alphabet = "ABCDE"
        const_idx = 1
        
        for factor, mult in factors_list:
            f_deg = sp.degree(factor, var)
            for i in range(1, mult + 1):
                if f_deg == 1:
                    symbol = sp.Symbol(f"A_{const_idx}")
                    partial_terms.append(symbol / (factor**i))
                    unknowns.append(symbol)
                    const_idx += 1
                else:
                    # Fattore quadratico irriducibile
                    symbol_a = sp.Symbol(f"B_{const_idx}")
                    symbol_b = sp.Symbol(f"C_{const_idx}")
                    partial_terms.append((symbol_a * var + symbol_b) / (factor**i))
                    unknowns.extend([symbol_a, symbol_b])
                    const_idx += 1

        decomp_form = sp.Add(*partial_terms, evaluate=False)
        # Nota: dobbiamo dividere per coeff_lead se presente per coerenza con D(x) = coeff_lead * factors
        # Ma solitamente è meglio normalizzare D(x) all'inizio o moltiplicare LHS per coeff_lead.
        # Per semplicità, consideriamo che D(x) sia già quello usato nei denominatori dei fratti.
        
        latex_steps.append({
            "title": "Forma della decomposizione",
            "content": rf"\frac{{{to_latex(R)}}}{{{to_latex(D_expr)}}} = {to_latex(decomp_form)}"
        })

        # PASSO 4: Calcolo dei coefficienti
        # R(x) = Sum( term_i * D(x) )
        # Dobbiamo assicurarci che term_i * D(x) sia semplificato correttamente
        rhs_expanded_sum = 0
        for term in partial_terms:
            rhs_expanded_sum += sp.simplify(term * D_expr)
        
        rhs_expanded = sp.expand(rhs_expanded_sum)
        
        # Sistema di equazioni
        system = []
        # Prendiamo tutte le potenze fino al grado di D-1
        for i in range(deg_D):
            # Coefficiente di var^i in R(x)
            lhs_c = sp.Poly(R, var).coeff_monomial(var**i)
            # Coefficiente di var^i in RHS
            rhs_c = sp.Poly(rhs_expanded, var).coeff_monomial(var**i)
            system.append(sp.Eq(lhs_c, rhs_c))
        
        sol = sp.solve(system, unknowns)
        
        # Preparazione stringhe per il sistema e la soluzione
        # Rimuoviamo equazioni banali 0 = 0
        filtered_system = [eq for eq in system if not (eq.lhs == 0 and eq.rhs == 0)]
        system_latex = rf"\begin{{cases}} " + r" \\ ".join([to_latex(eq) for eq in filtered_system]) + r" \end{{cases}}"
        
        if isinstance(sol, list): # Può succedere se ci sono infinite soluzioni o nessuna, ma qui dovrebbe essere un dict
            sol = sol[0] if sol else {}
            
        sol_latex = r", \ ".join([rf"{to_latex(u)} = {to_latex(sol.get(u, 0))}" for u in unknowns])

        latex_steps.append({
            "title": "Calcolo dei coefficienti",
            "content": rf"{to_latex(R)} = {to_latex(rhs_expanded)} \\ \text{{Confrontando i coefficienti:}} \\ {system_latex} \implies {sol_latex}"
        })

        # PASSO 5: Risultato finale
        # Usiamo sympy.apart per il risultato finale pulito (verifica)
        # Ma sostituiamo nella nostra forma per coerenza
        final_decomp = decomp_form.subs(sol)
        final_result = Q + final_decomp
        
        latex_steps.append({
            "title": "Risultato finale",
            "content": rf"\frac{{{to_latex(N_expr)}}}{{{to_latex(D_expr)}}} = {to_latex(final_result)}",
            "type": "final"
        })

        return jsonify({"success": True, "latex": latex_steps})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
