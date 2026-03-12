from flask import Blueprint, request, jsonify
import sympy as sp

exp_at_bp = Blueprint("exp_at", __name__)

@exp_at_bp.route("/api/expAt", methods=["POST"])
def compute_exp_At():
    t = sp.symbols("t")
    A = sp.Matrix(request.json["matrix"])
    N = sp.sympify(request.json["N_matrix_raw"])
    proj_expr_list = [sp.sympify(e) for e in request.json["proj_expr_list_raw"]]
    proj_lines = request.json.get("latex_proj", [])

    n = A.shape[0]
    D = A - N

    # Indice di nilpotenza
    k = 1
    N_pow = N
    while k <= n + 5:
        if N_pow.equals(sp.zeros(*N.shape)):
            break
        k += 1
        N_pow = N_pow * N

    # Calcolo degli autovalori (usati solo per e^{Dt})
    eigvals = {}
    for i, Ei in enumerate(proj_expr_list):
        lam = (Ei * D).trace() / Ei.trace()
        eigvals[sp.simplify(lam)] = 1

    # Costruzione e^{Dt}
    exp_Dt = sp.zeros(n)
    for i, (lam, _) in enumerate(eigvals.items()):
        base = sp.exp(sp.sympify(f"({lam})") * t, evaluate=False)
        exp_Dt += base * proj_expr_list[i]

    latex_Dt_formula = " + ".join([
        f"e^{{{sp.latex(lam)} t}} E_{{{i+1}}}" for i, (lam, _) in enumerate(eigvals.items())
    ])

    latex_matrix_rows = [
        " & ".join([sp.latex(cell) for cell in row])
        for row in exp_Dt.tolist()
    ]
    latex_Dt_matrix = "\\\\ \n".join(latex_matrix_rows)

    # Calcolo e^{At} = sum_{i=0}^{k-1} (t^i / i!) * e^{Dt} * N^i
    exp_At = sp.zeros(n)
    for i in range(k):
        scalar = sp.Pow(t, i) / sp.factorial(i)
        N_term = sp.simplify(N**i)
        exp_At += scalar * exp_Dt * N_term

    latex_exp = sp.latex(sp.simplify(exp_At))

    return jsonify({
        "success": True,
        "latex_exp": latex_exp,
        "latex_N": sp.latex(N),
        "latex_proj": proj_lines,
        "nilpotence_index": k,
        "latex_Dt_formula": latex_Dt_formula,
        "latex_Dt_matrix": f"\\left[\\begin{{matrix}}{latex_Dt_matrix}\\end{{matrix}}\\right]"
    })