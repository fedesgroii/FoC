from flask import Blueprint, request, jsonify
import sympy as sp
import re
from .utils import transformations_basic as transformations
from sympy.parsing.sympy_parser import parse_expr

da_soluzione_a_sistema_bp = Blueprint("da_soluzione_a_sistema", __name__)

def to_latex(expr):
    return sp.latex(expr)

def normalize_exponential(expr_str):
    """
    Converte e^(...) in exp(...)
    Mantiene compatibilità con exp(...) già presente
    """
    if not isinstance(expr_str, str):
        return expr_str
    
    # e^(...) → exp(...)
    expr_str = re.sub(r'e\^\(([^)]+)\)', r'exp(\1)', expr_str)
    # e**(...) → exp(...)
    expr_str = re.sub(r'e\*\*\(([^)]+)\)', r'exp(\1)', expr_str)
    
    return expr_str

def fix_latex_y_symbols(latex_str, time_type, n, is_derivative=True):
    # Replaces y_0, y_1... with proper derivatives or time shifts
    res = latex_str
    time_var = 't'
    for i in reversed(range(n + 1)):
        if time_type == 'continuous':
            if i == 0:
                res = res.replace(f"y_{{{i}}}", f"y({time_var})")
                res = res.replace(f"y_{i}", f"y({time_var})")
            elif i == 1:
                res = res.replace(f"y_{{{i}}}", f"\\dot{{y}}({time_var})")
                res = res.replace(f"y_{i}", f"\\dot{{y}}({time_var})")
            elif i == 2:
                res = res.replace(f"y_{{{i}}}", f"\\ddot{{y}}({time_var})")
                res = res.replace(f"y_{i}", f"\\ddot{{y}}({time_var})")
            else:
                res = res.replace(f"y_{{{i}}}", f"y^{{({i})}}({time_var})")
                res = res.replace(f"y_{i}", f"y^{{({i})}}({time_var})")
        else:
            if i == 0:
                res = res.replace(f"y_{{{i}}}", f"y({time_var})")
                res = res.replace(f"y_{i}", f"y({time_var})")
            else:
                res = res.replace(f"y_{{{i}}}", f"y({time_var}+{i})")
                res = res.replace(f"y_{i}", f"y({time_var}+{i})")
    return res

@da_soluzione_a_sistema_bp.route('/api/da_soluzione_a_sistema', methods=['POST'])
def api_da_soluzione_a_sistema():
    data = request.get_json()
    y_input = data.get("soluzione", "")
    time_type = data.get("time_type", "continuous") # continuous or discrete

    if not y_input.strip():
        return jsonify({"success": False, "error": "Nessuna soluzione fornita."})

    try:
        # Pre-process input
        y_input_clean = y_input.replace('^', '**')
        # find constants c1, c2, C1, C2, etc
        constants_str_found = list(set(re.findall(r'[cC]\d+', y_input_clean)))
        constants_str = sorted([c.lower() for c in constants_str_found], key=lambda x: int(x[1:]))
        # Ensure we only have unique lowercase constants
        constants_str = list(dict.fromkeys(constants_str))
        n = len(constants_str)

        if n == 0:
            return jsonify({"success": False, "error": "Nessuna costante c_i trovata nella soluzione. Assicurati di usare il formato c1, c2, ecc."})

        # Replace all uppercase C with lowercase c in the input string to match our symbols
        y_input_clean = re.sub(r'C(\d+)', r'c\1', y_input_clean)

        # Parse expression
        time_var_name = 't'
        if time_type == 'continuous':
            t_sym = sp.symbols('t', real=True)
        else:
            t_sym = sp.symbols('t', integer=True)
            
        local_dict = {time_var_name: t_sym, 'e': sp.exp(1)}
        for c in constants_str:
            local_dict[c] = sp.symbols(c)

        # Parsing dell'espressione
        y_expr = parse_expr(y_input_clean, local_dict=local_dict, transformations=transformations)
        y_expr = sp.expand(y_expr)
        
        latex_steps = []

        # 0. Sistema inserito
        time_domain_str = "Tempo Continuo (R)" if time_type == 'continuous' else "Tempo Discreto (Z)"
        latex_steps.append({
            "title": f"Soluzione Inserita - {time_domain_str}",
            "content": f"y({time_var_name}) = {to_latex(y_expr)}"
        })

        # 1. Ordine n
        latex_steps.append({
            "title": "Ordine del sistema n",
            "content": f"n = {n} \\quad \\text{{(Costanti: }} {', '.join(constants_str)} \\text{{)}}"
        })

        # 2. Generazione Equazioni e Derivate/Incrementi
        equations = [y_expr]
        for i in range(1, n + 1):
            if time_type == 'continuous':
                equations.append(sp.diff(equations[-1], t_sym))
            else:
                equations.append(y_expr.subs(t_sym, t_sym + i))
        
        y_syms = [sp.symbols(f"y_{i}") for i in range(n + 1)]
        
        eqs_latex = []
        for i in range(n + 1):
            eq_str = f"y_{i} = {to_latex(equations[i])}"
            eqs_latex.append(fix_latex_y_symbols(eq_str, time_type, n))
            
        latex_steps.append({
            "title": "Equazioni" if time_type == 'continuous' else "Incrementi",
            "content": r"\begin{aligned} " + " \\\\ ".join(eqs_latex) + r" \end{aligned}"
        })

        # 3. Sistema in forma matriciale
        C_syms = [local_dict[c] for c in constants_str]

        A = sp.zeros(n, n)
        b = sp.zeros(n, 1)

        for i in range(n):
            eq = equations[i]
            for j, c_sym in enumerate(C_syms):
                # Uso diff invece di coeff per maggiore robustezza
                A[i, j] = sp.simplify(eq.diff(c_sym))
            
            known_terms = eq
            for c_sym in C_syms:
                known_terms = known_terms.subs(c_sym, 0)
            
            b[i, 0] = sp.simplify(y_syms[i] - known_terms)

        c_syms_vec = sp.Matrix([c_sym for c_sym in C_syms])
        
        # FIX IMPAGINAZIONE SEZIONE 3
        # Separo la definizione della forma compatta dalla visualizzazione delle matrici
        step3_content = (
            r"\vec{b} = A \cdot \vec{c} \\ "
            r"{to_latex(b)} = {to_latex(A)} \cdot {to_latex(c_syms_vec)}"
        ).replace("{to_latex(b)}", to_latex(b)).replace("{to_latex(A)}", to_latex(A)).replace("{to_latex(c_syms_vec)}", to_latex(c_syms_vec))
        
        latex_steps.append({
            "title": "Sistema in forma matriciale",
            "content": fix_latex_y_symbols(step3_content, time_type, n)
        })

        # 4. Calcolo del determinante
        det_A = sp.simplify(A.det())
        det_is_zero = (sp.simplify(det_A) == 0)
        
        color = "green" if not det_is_zero else "red"
        # FIX IMPAGINAZIONE SEZIONE 4
        det_msg = "\\det(A) \\neq 0 \\rightarrow \\text{matrice invertibile}" if not det_is_zero else "\\det(A) = 0 \\rightarrow \\text{matrice singolare}"
        
        latex_steps.append({
            "title": "Calcolo del determinante",
            "content": (
                r"\det(A) = " + to_latex(det_A) + r" \\ " +
                r"\textcolor{" + color + r"}{" + det_msg + r"}"
            )
        })
        
        if det_is_zero:
            return jsonify({"success": False, "error": "Impossibile isolare le costanti. Il determinante della matrice A è zero (costanti linearmente dipendenti)."})

        # 5. Matrice inversa
        cofactor_matrix = A.cofactor_matrix()
        adj_A = sp.simplify(cofactor_matrix.T)
        A_inv = sp.simplify(adj_A / det_A)

        # FIX IMPAGINAZIONE SEZIONE 5
        # Uso \\\\ per andare a capo tra le matrici
        latex_steps.append({
            "title": "Matrice inversa",
            "content": (
            r"\begin{align*}" +
            r"&\text{Matrice dei cofattori: } \mathrm{Cof}(A) = \\ &" + to_latex(cofactor_matrix) + r"\\[2em]" +
            r"&\text{Matrice aggiunta: } \mathrm{Adj}(A) = \mathrm{Cof}(A)^T = \\ &" + to_latex(adj_A) + r"\\[2em]" +
            r"&A^{-1} = \frac{1}{\det(A)} \cdot \mathrm{Adj}(A) = \\ &" + to_latex(A_inv) +
            r"\end{align*}"
    )
})



        # 6. Calcolo delle costanti
        c_vector = sp.simplify(A_inv * b)
        
        sol_dict = {}
        for i, c_sym in enumerate(C_syms):
            sol_dict[c_sym] = sp.simplify(c_vector[i])

        # FIX IMPAGINAZIONE SEZIONE 6 (CRITICO)
        # La matrice A_inv * b è troppo larga se messa su una riga.
        # Soluzione: mostro l'operazione e poi mostro il risultato vettore su più righe o compatto.
        
        # Mostro la moltiplicazione
        multiplication_str = (
            r"\vec{c} = A^{-1} \cdot \vec{b} \\ " +
            to_latex(c_syms_vec) + r" = " + to_latex(A_inv) + r" \cdot " + to_latex(b)
        )
        
        latex_steps.append({
            "title": "Calcolo delle costanti (Moltiplicazione)",
            "content": fix_latex_y_symbols(multiplication_str, time_type, n)
        })

        latex_steps.append({
            "title": "Risultato Vettore Costanti",
            "content": (
                to_latex(c_syms_vec) + r" = " + to_latex(c_vector)
            )
        })
        
        sol_dict = {}
        for i, c_sym in enumerate(C_syms):
            sol_dict[c_sym] = sp.simplify(c_vector[i])

        # 7. Costanti isolate
        sol_latex = []
        for c_sym in C_syms:
            if c_sym in sol_dict:
                sol_str = f"{to_latex(c_sym)} = {to_latex(sol_dict[c_sym])}"
                sol_latex.append(fix_latex_y_symbols(sol_str, time_type, n))
        
        latex_steps.append({
            "title": "Costanti isolate",
            "content": (
                r"\begin{align*}" + "\n" +
                " \\\\[2.5em] ".join([f"&{s}" for s in sol_latex]) + "\n" +
                r"\end{align*}"
            )
        })

        # 8. Equazione finale
        y_n_expr = equations[n].subs(sol_dict)
        y_n_expr = sp.simplify(sp.expand(y_n_expr))
        
        # Build the final equation y_n = ...
        final_eq_latex = fix_latex_y_symbols(f"y_{n} = {to_latex(y_n_expr)}", time_type, n)
        
        final_eq_html = f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #f0f4f8 100%);
            border: 3px solid #007bff;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <h3 style="margin-top: 0; color: #007bff; font-size: 1.3rem; text-transform: uppercase; letter-spacing: 1px;">Risultato Finale</h3>
            <div style="font-size: 2.2rem; font-weight: bold; color: #212529; overflow-x: auto; padding: 10px 0;">
                \\( {final_eq_latex} \\)
            </div>
        </div>
        """
        
        latex_steps.append({
            "title": "Equazione differenziale finale" if time_type == 'continuous' else "Equazione alle differenze finale",
            "content": final_eq_html,
            "is_html": True
        })

        # 9. Variabili di Stato
        state_vars = []
        for i in range(n):
            if time_type == 'continuous':
                if i == 0:
                    state_vars.append(f"x_{{{i+1}}}({time_var_name}) = y({time_var_name})")
                elif i == 1:
                    state_vars.append(f"x_{{{i+1}}}({time_var_name}) = \\dot{{y}}({time_var_name})")
                elif i == 2:
                    state_vars.append(f"x_{{{i+1}}}({time_var_name}) = \\ddot{{y}}({time_var_name})")
                else:
                    state_vars.append(f"x_{{{i+1}}}({time_var_name}) = y^{{({i})}}({time_var_name})")
            else:
                if i == 0:
                    state_vars.append(f"x_{{{i+1}}}({time_var_name}) = y({time_var_name})")
                else:
                    state_vars.append(f"x_{{{i+1}}}({time_var_name}) = y({time_var_name}+{i})")
        
        latex_steps.append({
            "title": "Variabili di stato introdotte",
            "content": r"\begin{aligned} " + " \\ \\ \\ ".join(state_vars) + r" \end{aligned}"
        })

        # 10. Matrici Finali
        coeffs = [sp.simplify(y_n_expr.diff(y_syms[i])) for i in range(n)]
        u_term = sp.simplify(y_n_expr - sum(coeffs[i]*y_syms[i] for i in range(n)))
        
        A_comp = sp.zeros(n, n)
        for i in range(n - 1):
            A_comp[i, i + 1] = 1
        for i in range(n):
            A_comp[n - 1, i] = coeffs[i]
            
        B_comp = sp.zeros(n, 1)
        if u_term != 0:
            B_comp[n - 1, 0] = 1 # Assuming input u(t) acts on the last state variable
        
        C_comp = sp.zeros(1, n)
        C_comp[0, 0] = 1
        
        D_comp = sp.zeros(1, 1) # D is 0

        # Create vertical layout for matrices with clear labels and increased size
        html_content = f"""
        <div style="display: flex; flex-direction: column; gap: 25px; align-items: center; margin-top: 25px; width: 100%;">
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; width: 95%; max-width: 700px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 6px solid #28a745;">
                <div style="font-weight: bold; color: #198754; margin-bottom: 15px; font-size: 1.2rem; display: flex; align-items: center;">
                    <span style="background: #28a745; color: white; border-radius: 4px; padding: 2px 8px; margin-right: 10px; font-size: 0.9rem;">A</span> 
                    Matrice di Stato (Dinamica)
                </div>
                <div style="font-size: 2rem; overflow-x: auto; padding: 15px 0; text-align: center; color: #212529;">
                    \\( A = {to_latex(A_comp)} \\)
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; width: 95%; max-width: 700px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 6px solid #007bff;">
                <div style="font-weight: bold; color: #007bff; margin-bottom: 15px; font-size: 1.2rem; display: flex; align-items: center;">
                    <span style="background: #007bff; color: white; border-radius: 4px; padding: 2px 8px; margin-right: 10px; font-size: 0.9rem;">B</span> 
                    Matrice degli Ingressi (Controllo)
                </div>
                <div style="font-size: 2rem; overflow-x: auto; padding: 15px 0; text-align: center; color: #212529;">
                    \\( B = {to_latex(B_comp)} \\)
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; width: 95%; max-width: 700px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 6px solid #fd7e14;">
                <div style="font-weight: bold; color: #fd7e14; margin-bottom: 15px; font-size: 1.2rem; display: flex; align-items: center;">
                    <span style="background: #fd7e14; color: white; border-radius: 4px; padding: 2px 8px; margin-right: 10px; font-size: 0.9rem;">C</span> 
                    Matrice delle Uscite (Osservazione)
                </div>
                <div style="font-size: 2rem; overflow-x: auto; padding: 15px 0; text-align: center; color: #212529;">
                    \\( C = {to_latex(C_comp)} \\)
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; width: 95%; max-width: 700px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 6px solid #6c757d;">
                <div style="font-weight: bold; color: #495057; margin-bottom: 15px; font-size: 1.2rem; display: flex; align-items: center;">
                    <span style="background: #6c757d; color: white; border-radius: 4px; padding: 2px 8px; margin-right: 10px; font-size: 0.9rem;">D</span> 
                    Matrice di Legame Diretto
                </div>
                <div style="font-size: 2rem; overflow-x: auto; padding: 15px 0; text-align: center; color: #212529;">
                    \\( D = {to_latex(D_comp)} \\)
                </div>
            </div>
        </div>
        """
        
        if u_term != 0:
            html_content += f"<div style='text-align:center; margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 8px; color: #004085; font-weight: 500;'>Ingresso calcolato: \\( u({time_var_name}) = {to_latex(u_term)} \\)</div>"
        else:
            html_content += f"<div style='text-align:center; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; color: #6c757d; font-style: italic;'>Sistema autonomo (ingresso nullo)</div>"

        latex_steps.append({
            "title": "Rappresentazione in Spazio di Stato (Forma Compagna)",
            "content": html_content,
            "is_html": True
        })

        return jsonify({"success": True, "latex": latex_steps})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})