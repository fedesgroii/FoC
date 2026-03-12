from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import Matrix
from ast import literal_eval
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, parse_expr
from sympy.parsing.sympy_parser import convert_xor

sistemi_dinamici_bp = Blueprint("sistemi_dinamici", __name__)

@sistemi_dinamici_bp.route("/api/uscita_forzata", methods=["POST"])
def compute_output_y():
    t, s = sp.symbols("t s")
    A = sp.Matrix(request.json["A"])
    B = sp.Matrix(request.json["B"])
    C = sp.Matrix(request.json["C"]).reshape(1, A.shape[0])
    D = sp.sympify(request.json["D"])
    u_expr = request.json["u"] if "u" in request.json and request.json["u"].strip() else "0"
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    u = parse_expr(u_expr, local_dict={"t": t}, transformations=transformations)

    # Step 1: U(s) = Laplace{u(t)}
    U_s = sp.laplace_transform(u, t, s, noconds=True)
    if isinstance(U_s, tuple):
        U_s = U_s[0]

    # Step 2: H(s) = C (sI - A)^(-1)
    I = sp.eye(A.shape[0])
    H_s = (C * (s * I - A).inv()).reshape(1, A.shape[0])

    # Step 3: W(s) = H(s)B + D
    W_s = H_s * B + sp.Matrix([[D]])

    # Step 4: y(s) = H(s)x0 + W(s)U(s) — only W(s)U(s) part for now, x0 will be zero
    y_s = (W_s * U_s)[0].expand()

    latex_steps = [
        {
            "title": "Trasformata dell'ingresso:",
            "content": f"U(s) = \\mathcal{{L}}\\{{u(t)\\}} = {sp.latex(U_s)}"
        },
        {
            "title": "Calcolo della funzione di trasferimento:",
            "content": f"H(s) =C (sI - A)^{{-1}} = {sp.latex(H_s.applyfunc(sp.factor))}"
        },
        {
            "title": "Calcolo della funzione d'ingresso moltiplicata:",
            "content": f"W(s) = H(s) B + D = {sp.latex(sp.factor((H_s * B)[0]))} + {sp.latex(D)}"
        },
        {
            "title": "Calcolo dell'uscita nel dominio di Laplace:",
            "content": f"\\text{{Poiché }} x_0 = 0, \\quad y(s) = W(s) U(s) = {' + '.join([sp.latex(sp.factor(term)) for term in y_s.as_ordered_terms()])}"
        },
        {
            "title": "Inversione della trasformata di Laplace:",
            "content": f"y(t) = \\mathcal{{L}}^{{-1}}\\{{y(s)\\}} = {sp.latex(sp.inverse_laplace_transform(y_s, s, t).subs(sp.Heaviside(t), 1).subs(sp.Heaviside(t - 0), 1).expand())}"
        }
    ]
    return jsonify({"latex": latex_steps})