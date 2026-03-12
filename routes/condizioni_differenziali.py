from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import symbols, Function, Eq, Derivative, dsolve, simplify, latex
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from routes.equazioni_differenziali import solve_differential_equation

condizioni_differenziali_bp = Blueprint("condizioni_differenziali", __name__)

@condizioni_differenziali_bp.route("/api/condizioni_differenziali", methods=["POST"])
def api_condizioni_iniziali():
    data = request.get_json()

    # Recupera l'equazione e le condizioni iniziali
    eq_str = data.get("equazione", "")
    condizioni = data.get("condizioniIniziali", [])

    if not eq_str.strip():
        return jsonify({
            "success": False,
            "errore": "Nessuna equazione differenziale fornita."
        })

    try:
        t = symbols("t")
        y = Function("y")(t)

        transformations = standard_transformations + (implicit_multiplication_application,)
        local_dict = {"y": y, "t": t, "e": sp.E}

        lhs_str, rhs_str = eq_str.split("=", 1)
        lhs_cleaned = lhs_str.strip().replace("^", "**")
        rhs_cleaned = rhs_str.strip().replace("^", "**")
        lhs_expr = parse_expr(lhs_cleaned, transformations=transformations, local_dict=local_dict)
        rhs_expr = parse_expr(rhs_cleaned, transformations=transformations, local_dict=local_dict)
        print("[DEBUG] LHS parsed:", lhs_expr)
        print("[DEBUG] RHS parsed:", rhs_expr)

        def convert_d_operator(expr):
            if not expr.has(sp.Symbol('d')):
                return expr
            expr = sp.expand(expr)
            d = sp.Symbol('d')
            result = 0
            for term in expr.as_ordered_terms():
                if term.has(y):
                    coeffs = term.as_coeff_mul()
                    base_factors = coeffs[1]
                    power = 0
                    for f in base_factors:
                        if f == d:
                            power += 1
                        elif isinstance(f, sp.Pow) and f.base == d:
                            power += f.exp
                    if power > 0:
                        new_term = coeffs[0] * Derivative(y, (t, power))
                        result += new_term
                    else:
                        result += term
                else:
                    result += term
            return result

        lhs_expr = convert_d_operator(lhs_expr)
        rhs_expr = convert_d_operator(rhs_expr)
        print("[DEBUG] LHS dopo conversione d:", lhs_expr)
        print("[DEBUG] RHS dopo conversione d:", rhs_expr)

        lhs_expr = sp.expand(lhs_expr)
        rhs_expr = sp.expand(rhs_expr)

        if not any(lhs_expr.has(Derivative(y, (t, n))) for n in range(1, 11)):
            raise ValueError("L'equazione deve contenere almeno una derivata di y(t)")

        original_eq = Eq(lhs_expr, rhs_expr)
        print("[DEBUG] Equazione completa:", original_eq)


        latex_steps = []

        # Calcola il grado dell'equazione differenziale rispetto a y
        grado = 0
        for deriv in lhs_expr.atoms(Derivative):
            if deriv.expr == y:
                ordine = deriv.derivative_count
                grado = max(grado, ordine)

        # Verifica che il numero di condizioni iniziali corrisponda
        if len(condizioni) != grado:
            return jsonify({
                "success": False,
                # Messaggio HTML con delimitatori LaTeX per MathJax
                "errore": (
                    '<p style="color:red; text-align:center;">'
                    f'Hai fornito {len(condizioni)} condizioni iniziali, ma il polinomio '
                    f'\\(P(\\Delta)\\) è di grado {grado}.'
                    '</p>'
                )
            })

        # Inserisce lo step finale dell'equazione originale (senza condizioni)
        soluzione_generale = solve_differential_equation(eq_str)
        if soluzione_generale.get("success") and isinstance(soluzione_generale.get("latex"), list):
            step_finale = soluzione_generale["latex"][-1]
            latex_steps.append({
                "title": "Soluzione generale dell'equazione:",
                "content": step_finale["content"]
            })

        try:
            general_sol = dsolve(original_eq, y)
            if isinstance(general_sol, list):
                y_expr = general_sol[0].rhs
            else:
                y_expr = general_sol.rhs

            eqs_valutate = []

            # Rinomina simboli C_i in c_i anche nel sistema non ancora risolto
            rename_dict = {C: sp.Symbol(f'c_{i+1}') for i, C in enumerate(sorted([s for s in y_expr.free_symbols if s.name.startswith("C")], key=lambda s: s.name))}
            # Costruisci equazioni con condizioni
            for i in range(grado):
                deriv = y_expr.diff(t, i).subs(t, 0)
                rhs = sp.sympify(condizioni[i])
                eq = sp.Eq(deriv, rhs)
                eqs_valutate.append(eq)
            eqs_valutate = [eq.xreplace(rename_dict) for eq in eqs_valutate]

            latex_eqs = "\\\\\n".join([sp.latex(eq) for eq in eqs_valutate])

            soluzioni = sp.solve(eqs_valutate, *rename_dict.values(), dict=True)
            if soluzioni:
                soluzioni_eqs = [sp.Eq(var, simplify(expr)) for var, expr in soluzioni[0].items()]
                latex_soluzioni = "\\\\\n".join([sp.latex(eq) for eq in soluzioni_eqs])
                sistema_totale = rf"\begin{{cases}} {latex_eqs} \end{{cases}} \quad \Rightarrow \quad \begin{{cases}} {latex_soluzioni} \end{{cases}}"
            else:
                sistema_totale = rf"\begin{{cases}} {latex_eqs} \end{{cases}} \quad \Rightarrow \quad \text{{nessuna soluzione trovata}}"

            latex_steps.append({
                "title": "Sistema delle condizioni applicate:",
                "content": sistema_totale
            })
        except Exception as e:
            latex_steps.append({
                "title": "Errore durante la costruzione del sistema",
                "content": f"Impossibile derivare la soluzione: {str(e)}"
            })

        # Applica le soluzioni simboliche alla forma generale
        if soluzioni:
            sostituzioni = {k: v for k, v in soluzioni[0].items()}
            y_expr = y_expr.xreplace(rename_dict)  # sostituisci C_i → c_i
            y_finale = y_expr.subs(sostituzioni)
            y_finale = sp.expand(y_finale)

            # Estrai la parte libera (che contiene y_0, y_1, ...) e la parte forzata (tutto il resto)
            simboli_libera = [sp.Symbol(f'y_{i}') for i in range(grado)]
            # Costruisci parte libera raccogliendo tutti i termini che dipendono da y_i
            parte_libera = sum(
                term for term in (y_finale.args if y_finale.is_Add else [y_finale])
                if any(sym in term.free_symbols for sym in simboli_libera)
            )
            # Il resto è forzato
            parte_forzata = y_finale - parte_libera

            latex_steps.append({
                "title": "Soluzione complessiva:",
                "content": rf"y(t) = \underbrace{{{sp.latex(parte_libera)}}}_\text{{risposta libera}} + \underbrace{{{sp.latex(parte_forzata)}}}_\text{{risposta forzata}}"
            })

        return jsonify({
            "success": True,
            "latex": latex_steps
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "errore": f"Errore imprevisto: {str(e)}"
        })