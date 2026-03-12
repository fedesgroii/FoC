from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import symbols, Function, Eq, simplify, latex
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

condizioni_alle_differenze_bp = Blueprint("condizioni_alle_differenze", __name__)

@condizioni_alle_differenze_bp.route("/api/condizioni_alle_differenze", methods=["POST"])
def api_condizioni_differenze():
    data = request.get_json()
    print("[DEBUG] Payload ricevuto:", data)
    try:
        eq_str = data.get("equazione", "")
        condizioni = data.get("condizioniIniziali", [])

        latex_steps = []

        t = symbols("t")
        y = Function("y")
        Y = sp.Symbol("y")  # y come simbolo temporaneo per parsing
        transformations = standard_transformations + (implicit_multiplication_application,)
        local_dict = {"y": Y, "t": t, "e": sp.E, "delta": Function("delta")}

        lhs_str, rhs_str = eq_str.split("=", 1)
        lhs_expr = parse_expr(lhs_str.strip().replace("^", "**"), transformations=transformations, local_dict=local_dict)
        lhs_expr = lhs_expr.subs(Y, y(t))
        print("[DEBUG] lhs_expr iniziale:", lhs_expr)
        print("[DEBUG] lhs_expr type:", type(lhs_expr))
        rhs_expr = parse_expr(rhs_str.strip().replace("^", "**"), transformations=transformations, local_dict=local_dict)
        rhs_expr = rhs_expr.subs(Y, y(t))
        print("[DEBUG] rhs_expr iniziale:", rhs_expr)
        print("[DEBUG] rhs_expr type:", type(rhs_expr))
        print("[DEBUG] lhs_expr dopo correzione:", lhs_expr)
        print("[DEBUG] rhs_expr dopo correzione:", rhs_expr)

        # Calcola il grado del polinomio alle differenze come grado del polinomio in d
        delta = sp.Symbol('d')
        lhs_poly = lhs_expr.as_poly(delta)
        grado = lhs_poly.degree() if lhs_poly is not None else 0

        if len(condizioni) != grado:
            return jsonify({
                "success": False,
                "errore": (
                    '<p style="color:red; text-align:center;">'
                    f'Hai fornito {len(condizioni)} condizioni iniziali, ma il polinomio '
                    f'\\(P(\\Delta)\\) è di grado {grado}.'
                    '</p>'
                )
            })

        soluzione_symbolica = data.get("soluzione_symbolica", "")
        if not soluzione_symbolica:
            print("[DEBUG] Soluzione simbolica assente nel payload.")
            return jsonify({
                "success": False,
                "errore": "Soluzione simbolica mancante dal payload della richiesta."
            })
        print("[DEBUG] Soluzione simbolica ricevuta:", soluzione_symbolica)
        y_expr = parse_expr(soluzione_symbolica, local_dict={"t": t, "y": y, "delta": Function("delta")})
        print("[DEBUG] y_expr ottenuta dall'API:", y_expr)
        # Aggiunge step per la soluzione generale in LaTeX
        from sympy import latex as _latex
        latex_steps.append({
            "title": "Soluzione generale dell'equazione:",
            "content": f"y(t) = {_latex(y_expr)}"
        })

        rename_dict = {C: sp.Symbol(f'c_{i+1}') for i, C in enumerate(sorted([s for s in y_expr.free_symbols if s.name.startswith("C")], key=lambda s: s.name))}

        eqs_valutate = []
        for k, val in enumerate(condizioni):
            # Applica la definizione discreta del delta di Dirac: delta(t - k) = 1 se t == k, 0 altrimenti
            def delta_discreta(expr):
                return expr.replace(
                    lambda f: f.func.__name__ == "delta",
                    lambda f: sp.Integer(1) if f.args[0] == 0 else sp.Integer(0)
                )

            lhs = y_expr.subs(t, k)
            lhs = lhs.replace(Function("delta"), lambda arg: sp.Piecewise((1, arg == 0), (0, True)))
            lhs = sp.simplify(lhs)
            rhs = sp.sympify(val)
            eq = sp.Eq(lhs, rhs)
            eqs_valutate.append(eq)

        print("[DEBUG] Equazioni del sistema (y(k) = y_k):")
        for eq in eqs_valutate:
            print("   ", eq)

        latex_eqs = "\\\\\n".join([sp.latex(eq) for eq in eqs_valutate])

        soluzioni = sp.solve(eqs_valutate, *rename_dict.values(), dict=True)
        print("[DEBUG] Soluzioni trovate:", soluzioni)
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

        if soluzioni:
            sostituzioni = {k: v for k, v in soluzioni[0].items()}
            y_finale = y_expr.subs(sostituzioni)
            print("[DEBUG] y(t) con costanti trovate:", y_finale)
            y_finale = sp.expand(y_finale)

            simboli_libera = [sp.Symbol(f'y_{i}') for i in range(len(condizioni))]
            parte_libera = sum(
                term for term in (y_finale.args if y_finale.is_Add else [y_finale])
                if any(sym in term.free_symbols for sym in simboli_libera)
            )
            parte_forzata = y_finale - parte_libera

            latex_steps.append({
                "title": "Soluzione complessiva:",
                "content": rf"y(t) = \underbrace{{{sp.latex(parte_libera)}}}_\text{{risposta libera}} + \underbrace{{{sp.latex(parte_forzata)}}}_\text{{risposta forzata}}"
            })

        return jsonify({
            "success": True,
            "latex": latex_steps,
            "soluzione_symbolica": str(y_finale) if soluzioni else ""
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "errore": f"Errore imprevisto: {str(e)}"
        })