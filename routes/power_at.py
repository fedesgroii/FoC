from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import Matrix, symbols, simplify
from ast import literal_eval

power_at_bp = Blueprint("power_at", __name__)

@power_at_bp.route("/api/powerAt", methods=["POST"])
def compute_power_At():
    t = sp.symbols("t")
    A = sp.Matrix(request.json["matrix"])

    # Ricava N ed E_i dalla decomposizione spettrale precedente
    N = sp.sympify(request.json["N_matrix_raw"])
    proj_expr_list = [sp.sympify(e) for e in request.json["proj_expr_list_raw"]]

    n = A.shape[0]
    eigvals = A.eigenvals()
    proj_lines = request.json.get("latex_proj", [])

    # Calcolo D^t = sum λ_i^t E_i
    D_exp_formula_latex = " + ".join(
        [f"{sp.latex(lam)}^t E_{{{i+1}}}" for i, lam in enumerate(eigvals.keys())]
    )
    D_exp_value = sp.zeros(n)
    for i, (lam, _) in enumerate(eigvals.items()):
        D_exp_value += sp.Pow(sp.sympify(f"({lam})"), t, evaluate=False) * proj_expr_list[i]

    latex_matrix_rows = [
        " & ".join([sp.latex(cell) for cell in row])
        for row in D_exp_value.tolist()
    ]
    latex_Dt_matrix = "\\\\ \n".join(latex_matrix_rows)

    # Calcolo dell'indice di nilpotenza
    k = 1
    N_power = N
    while k <= n + 5:
        if N_power.equals(sp.zeros(*N.shape)):
            break
        k += 1
        N_power = N_power * N

    At_J = sp.zeros(n)
    for i in range(k):
        binom = sp.binomial(t, i)
        D_term = sp.zeros(n)
        for j, (lam, _) in enumerate(eigvals.items()):
            base_t_i = sp.Pow(sp.sympify(f"({lam})"), t - i, evaluate=False)
            D_term += base_t_i * proj_expr_list[j]
        N_term = simplify(N**i)
        term = simplify(binom * D_term * N_term)
        At_J += term

    At = simplify(At_J)
    latex_At = sp.latex(At)

    return jsonify({
        "success": True,
        "latex_power": latex_At,
        "nilpotence_index": k,
        "step_power": f"A^t = " + " + ".join([f"\\binom{{t}}{{{i}}} D^{{t-{i}}} N^{{{i}}}" for i in range(k)]),
        "latex_N": sp.latex(N),
        "latex_proj": proj_lines,
        "latex_Dt_formula": D_exp_formula_latex,
        "latex_Dt_matrix": f"\\left[\\begin{{matrix}}{latex_Dt_matrix}\\end{{matrix}}\\right]"
    })