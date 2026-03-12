from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import Function, symbols, Eq
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

equazioni_alle_differenze_bp = Blueprint("equazioni_alle_differenze", __name__)

@equazioni_alle_differenze_bp.route("/api/equazioni_alle_differenze", methods=["POST"])
def api_equazioni_alle_differenze():
    data = request.get_json()

    import re

    raw_poly = data.get("poly", "").strip()
    # Inserisce moltiplicazioni implicite mancanti, es: d(d-2) → d*(d-2)
    poly = re.sub(r'(\w)(\()', r'\1*(', raw_poly)
    poly = re.sub(r'\)(\w)', r')*\1', poly)
    poly = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', poly)
    rhs = data.get("rhs", "").strip()
    condizioni = data.get("condizioniIniziali", [])

    if not poly:
        return jsonify({
            "success": False,
            "errore": "Nessuna equazione alle differenze fornita."
        })

    eq_str = f"{poly} * y = {rhs}"

    try:
        t = sp.symbols("t", integer=True)
        y = Function("y")
        d = sp.Symbol("d")  # simbolo formale per rappresentare l'operatore shift

        transformations = standard_transformations + (implicit_multiplication_application,)
        local_dict = {"d": d, "y": y(t), "t": t, "e": sp.E}

        lhs_str, rhs_str = eq_str.split("=", 1)
        lhs_cleaned = lhs_str.strip().replace("^", "**")
        rhs_cleaned = rhs_str.strip().replace("^", "**")

        lhs_expr = parse_expr(lhs_cleaned, transformations=transformations, local_dict=local_dict)
        rhs_expr = parse_expr(rhs_cleaned, transformations=transformations, local_dict=local_dict)

        # DEBUG
        print("[DEBUG] LHS raw:", lhs_expr)
        print("[DEBUG] RHS raw:", rhs_expr)

        # Step 1: converte il polinomio in d in un'equazione in y(t+k)
        def convert_d_polynomial_to_difference(lhs_poly, y_func, t_var):
            d = sp.Symbol("d")
            terms = sp.expand(lhs_poly).as_ordered_terms()
            expr = 0
            for term in terms:
                coeff, rest = term.as_coeff_mul()
                power = 0
                for part in rest:
                    if part == d:
                        power += 1
                    elif isinstance(part, sp.Pow) and part.base == d:
                        power += part.exp
                expr += coeff * y_func(t_var + power)
            return expr

        lhs_final = convert_d_polynomial_to_difference(lhs_expr, y, t)
        rhs_final = rhs_expr

        equation = Eq(lhs_final, rhs_final)

        print("[DEBUG] Equazione completa:", equation)

        sol = sp.rsolve(equation, y(t))
        print("[DEBUG] Soluzione:", sol)

        # Aggiunge impulsi traslati se il polinomio è della forma d^n
        impulsi = []
        grado_impulsi = 0
        lhs_fattorizzato = sp.factor(lhs_expr)
        if isinstance(lhs_fattorizzato, sp.Mul):
            for factor in lhs_fattorizzato.args:
                if factor == d:
                    grado_impulsi += 1
                elif isinstance(factor, sp.Pow) and factor.base == d:
                    grado_impulsi += int(factor.exp)
        elif lhs_fattorizzato == d:
            grado_impulsi = 1
        elif isinstance(lhs_fattorizzato, sp.Pow) and lhs_fattorizzato.base == d:
            grado_impulsi = int(lhs_fattorizzato.exp)

        if grado_impulsi > 0:
            for k in range(grado_impulsi):
                delta_k = sp.Function("delta")(t - k)
                c_k = sp.Symbol(f"c_{k+1}")  # inizia da c_1 invece di c_0
                impulsi.append(c_k * delta_k)

            y_h = sum(impulsi)
            if sol is not None:
                sol += y_h
            else:
                sol = y_h

        # Rinomina simboli C_i → c_{i+grado_impulsi+1}
        rename_dict = {}
        for s in sol.free_symbols:
            if s.name.startswith("C"):
                try:
                    index = int(s.name[1:])
                    rename_dict[s] = sp.Symbol(f"c_{index + grado_impulsi + 1}")
                except ValueError:
                    continue

        sol = sol.xreplace(rename_dict)

        # Riordina i termini in base all'indice numerico del coefficiente c_i (impulsi prima, considera tutte le occorrenze di c_i)
        if sol.is_Add:
            delta_terms = []
            other_terms = []

            def indice(term):
                for s in term.free_symbols:
                    if s.name.startswith("c_") and "_" in s.name:
                        try:
                            return int(s.name.split("_")[1])
                        except ValueError:
                            continue
                return 0  # metti i delta senza coefficiente numerico esplicito all'inizio

            for term in sol.args:
                if any(f.func.__name__ == "delta" for f in term.atoms(sp.Function)):
                    delta_terms.append(term)
                else:
                    other_terms.append(term)

            delta_terms_sorted = sorted(delta_terms, key=indice)
            other_terms_sorted = sorted(other_terms, key=indice)
            sol = sp.Add(*delta_terms_sorted, *other_terms_sorted, evaluate=False)

        from sympy import latex

        # Trasforma termini del tipo (2i)^t o (-2i)^t in forma reale con seno/coseno
        from sympy import I, simplify, cos, sin, pi

        def convert_complex_exponentials(expr, t):
            if expr.is_Add:
                terms = expr.args
            else:
                terms = [expr]

            new_terms = []
            for term in terms:
                if term.has(sp.I) and term.has(sp.Pow):
                    c2 = term.coeff((2*sp.I)**t)
                    c3 = term.coeff((-2*sp.I)**t)
                    if c2 != 0 or c3 != 0:
                        replacement = (
                            -c2 * 2**t * cos(pi*t/2) +
                            c3 * 2**t * sin(pi*t/2)
                        )
                        new_terms.append(replacement)
                        continue
                new_terms.append(term)
            return sp.Add(*new_terms)

        # Applica la trasformazione alla soluzione completa
        sol = convert_complex_exponentials(sol, t)

        if sol.is_Add:
            terms = list(sol.args)
        else:
            terms = [sol]

        # Separa parte omogenea (con c_i) e particolare (senza c_i)
        parte_omogenea = []
        parte_particolare = []

        for term in terms:
            if any(s.name.startswith("c_") for s in term.free_symbols):
                parte_omogenea.append(term)
            else:
                parte_particolare.append(term)

        latex_omo = " + ".join([latex(term) for term in parte_omogenea]) or "0"
        latex_part = " + ".join([latex(term) for term in parte_particolare]) or "0"
        latex_tot = " + ".join([latex(term) for term in parte_omogenea + parte_particolare])

        soluzione_symbolica = str(sol)

        return jsonify({
            "success": True,
            "latex": [
                {
                    "title": "Soluzione omogenea:",
                    "content": f"y_o(t) = {latex_omo}"
                },
                {
                    "title": "Soluzione particolare:",
                    "content": f"y_p(t) = {latex_part}"
                },
                {
                    "title": "Soluzione generale:",
                    "content": f"y(t) = {latex_tot}"
                }
            ],
            "soluzione_symbolica": soluzione_symbolica
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "errore": f"Errore imprevisto durante il parsing: {str(e)}"
        })
