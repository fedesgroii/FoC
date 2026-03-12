from flask import Blueprint, request, jsonify
import sympy as sp
from sympy import Matrix

stabilita_bp = Blueprint("stabilita", __name__)

@stabilita_bp.route("/api/stabilita", methods=["POST"])
def compute_stabilita():
    try:
        data = request.json
        matrix_data = data.get("matrix")
        dominio = data.get("dominio") # 'R' (Continuo) o 'Z' (Discreto)
        
        A = sp.Matrix(matrix_data)
        n = A.shape[0]
        s = sp.symbols('s')
        
        # Polinomio caratteristico
        char_poly = A.charpoly(s).as_expr()
        factored = sp.factor(char_poly)
        
        # Autovalori e loro molteplicità algebrica e geometrica
        eigenvals_info = A.eigenvects()
        
        eig_list_latex = []
        is_stable = True
        is_unstable = False
        is_marginally_stable = False
        
        max_re = -float('inf')
        max_mod = -float('inf')
        
        marginally_stable_candidates_R = []
        marginally_stable_candidates_Z = []

        all_zeros = True

        for idx, (eigval, algebraic_mult, eigenvectors) in enumerate(eigenvals_info):
            geometric_mult = len(eigenvectors)
            
            # Semplifica l'autovalore per l'output in LaTeX
            # sympy.latex supporta i complessi
            eig_latex = sp.latex(eigval)
            
            # Formattazione per la lista degli autovalori
            if algebraic_mult > 1:
                eig_list_latex.append(rf"\lambda_{{{idx+1}}} = {eig_latex} \quad (\mu_a={algebraic_mult}, \mu_g={geometric_mult})")
            else:
                eig_list_latex.append(rf"\lambda_{{{idx+1}}} = {eig_latex}")
            
            # Calcolo stabilità in base al dominio
            val_complex = complex(eigval.evalf())
            re_part = val_complex.real
            im_part = val_complex.imag
            modulus = abs(val_complex)
            
            if abs(re_part) > 1e-9 or abs(im_part) > 1e-9:
                all_zeros = False

            if dominio == 'R':
                if re_part > max_re:
                    max_re = re_part
                if re_part > 1e-9: # > 0 considerata tolleranza numerica
                    is_unstable = True
                elif abs(re_part) <= 1e-9:
                    marginally_stable_candidates_R.append((algebraic_mult, geometric_mult))
            elif dominio == 'Z':
                if modulus > max_mod:
                    max_mod = modulus
                if modulus > 1 + 1e-9:
                    is_unstable = True
                elif abs(modulus - 1) <= 1e-9:
                    marginally_stable_candidates_Z.append((algebraic_mult, geometric_mult))

        # Regole Matematiche Rigorose
        instabile = False
        asintoticamente_stabile = True
        stabile_non_asintoticamente = False
        
        motivo_str = ""
        
        if dominio == 'R':
            # Tempo Continuo: dipendenza da a = Re(lambda)
            # 1. Asintoticamente Stabile: TUTTE le radici hanno a < 0
            # 2. Instabile: ALMENO UNA radice ha a > 0 OPPURE radici multiple con a = 0
            # 3. Stabile (ma non asintoticamente): Nessuna radice con a > 0 E radici semplici con a = 0
            for a, g in marginally_stable_candidates_R:
                if a > 1:
                    instabile = True
                    motivo_str = "presenza di radici multiple con parte reale nulla (che generano termini polinomiali divergenti)."
                    break

            if max_re > 1e-9:
                instabile = True
                motivo_str = "presenza di almeno un autovalore con parte reale positiva."
            
            if instabile:
                status_text = f"Instabile: {motivo_str}"
                status_color = "#e74c3c"
            elif max_re < -1e-9:
                status_text = "Asintoticamente Stabile: tutte le radici hanno parte reale negativa."
                status_color = "#2ecc71"
            else:
                status_text = "Stabile (ma non asintoticamente): nessuna radice con parte reale positiva e presenza di radici semplici con parte reale nulla."
                status_color = "#f39c12"

        elif dominio == 'Z':
            # Tempo Discreto: dipendenza da modulo = |lambda|
            # 1. Asintoticamente Stabile: TUTTE le radici hanno |lambda| < 1
            # 2. Instabile: ALMENO UNA radice ha |lambda| > 1 OPPURE radici multiple con |lambda| = 1
            # 3. Stabile (ma non asintoticamente): Nessuna radice con |lambda| > 1 E radici semplici con |lambda| = 1
            for a, g in marginally_stable_candidates_Z:
                if a > 1:
                    instabile = True
                    motivo_str = "presenza di radici multiple con modulo pari a 1 (che generano termini polinomiali divergenti)."
                    break

            if max_mod > 1 + 1e-9:
                instabile = True
                motivo_str = "presenza di almeno un autovalore con modulo maggiore di 1."

            if instabile:
                status_text = f"Instabile: {motivo_str}"
                status_color = "#e74c3c"
            elif max_mod < 1 - 1e-9:
                status_text = "Asintoticamente Stabile: tutte le radici hanno modulo minore di 1."
                status_color = "#2ecc71"
            else:
                status_text = "Stabile (ma non asintoticamente): nessuna radice con modulo > 1 e presenza di radici semplici con modulo pari a 1."
                status_color = "#f39c12"

        return jsonify({
            "success": True,
            "latex_char_poly": [
                rf"P(s) = {sp.latex(char_poly)}",
                rf"= {sp.latex(factored)}"
            ],
            "latex_eigenvalues": eig_list_latex,
            "stability_status": status_text,
            "stability_color": status_color
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
