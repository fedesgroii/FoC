from flask import request, jsonify
import sympy as sp

def compute():
    s = sp.symbols('s')
    A = sp.Matrix(request.json["matrix"])
    char_poly = A.charpoly(s).as_expr()
    factored = sp.factor(char_poly)
    eigvals = A.eigenvals()

    eig_list = [f"\\lambda_{{{i+1}}} = {sp.latex(val)}" for i, val in enumerate(eigvals.keys())]
    inv_P = sp.apart(1 / char_poly, s)

    inv_terms = sp.apart(1 / char_poly, s).as_ordered_terms()
    blocks = {}
    for t in inv_terms:
        denom = sp.denom(t)
        root = sp.solve(sp.Eq(denom, 0), s)[0]
        blocks.setdefault(root, []).append(t)

    latex_fi = []
    ei_list = []
    for idx, (lam, tl) in enumerate(blocks.items(), start=1):
        fi_expr = sp.simplify(sp.Add(*tl) * char_poly)
        denom = fi_expr.subs(s, lam)
        ei_expr = sp.simplify(fi_expr / denom)
        ei_list.append(ei_expr)
        latex_fi.append(rf"f_{idx}(s)=\dfrac{{{sp.latex(fi_expr)}}}{{P(s)}}")

    proj_lines = []
    proj_expr_list = []
    n = A.shape[0]
    I = sp.eye(n)

    for idx, ei_s in enumerate(ei_list, start=1):
        ei_poly = sp.Poly(sp.expand(ei_s), s)
        coeffs = list(reversed(ei_poly.all_coeffs()))
        expr_A = sp.zeros(n)
        latex_terms = []

        for k, coeff in enumerate(coeffs):
            coeff = sp.nsimplify(coeff, rational=True)
            if coeff == 0:
                continue
            sign_tex = "-" if coeff < 0 else ""
            abs_coeff = abs(coeff)
            coeff_tex = sp.latex(abs_coeff)
            factor_tex = "I" if k == 0 else "A" if k == 1 else f"A^{k}"
            term = I if k == 0 else A if k == 1 else A ** k
            expr_A += coeff * term
            if abs_coeff == 1:
                latex_terms.append(f"{sign_tex}{factor_tex}")
            else:
                latex_terms.append(f"{sign_tex}\\left({coeff_tex}\\right){factor_tex}")

        latex_expr_A = " + ".join(latex_terms).replace("+ -", "- ")
        Ei = expr_A
        proj_expr_list.append(Ei)
        proj_lines.append(rf"E_{idx} = e_{idx}(A) = {latex_expr_A} = {sp.latex(Ei)}")

    D_matrix = sp.zeros(n)
    for i, (lam, _) in enumerate(blocks.items(), start=1):
        Ei = proj_expr_list[i - 1]
        D_matrix += lam * Ei

    N_matrix = A - D_matrix
    latex_D = rf"D = {' + '.join([f'{sp.latex(lam)} E_{i+1}' for i, (lam, _) in enumerate(blocks.items())])} = {sp.latex(D_matrix)}"
    latex_N = rf"N = A - D = {sp.latex(N_matrix)}"

    return jsonify({
        "success": True,
        "latex": [
            rf"P(s) = {sp.latex(char_poly)}",
            rf"= {sp.latex(factored)}",
            r",\ ".join(eig_list),
            rf"\frac{{1}}{{P(s)}} = {sp.latex(inv_P)}"
        ],
        "latex_fi": latex_fi,
        "latex_proj": proj_lines,
        "latex_D": latex_D,
        "latex_N": latex_N,
        "N_matrix_raw": sp.srepr(N_matrix),
        "proj_expr_list_raw": [sp.srepr(ei) for ei in proj_expr_list]
    })