from flask import Blueprint, request, jsonify, render_template
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from .utils import transformations_basic as transformations
import traceback

delta_da_problema_a_soluzione_bp = Blueprint("delta_da_problema_a_soluzione", __name__)

def to_latex(expr):
    return sp.latex(expr)

class DeltaSpec:
    def __init__(self, time_expr_str, active_at):
        self.time_expr_str = time_expr_str
        self.active_at = int(active_at)
        self.constant = None
        self.found = False
        
        t = sp.symbols('t', real=True)
        local_dict = {'t': t}
        self.time_expr_sym = parse_expr(self.time_expr_str, local_dict=local_dict, transformations=transformations)

class ImpulseResponseSolver:
    def __init__(self):
        self.delta_specs = []
        self.constants = []
        self.equation_sym = None
        self.h_values = {}
        self.M = 0
        self.H_z = None
        self.N_z = None
        self.D_z = None
        self.n = 0
        self.D_val = None
        self.a_coeffs = []
        self.b_coeffs = []
        self.A = None
        self.B = None
        self.C = None
        self.D_mat = None
        self.steps_latex = []
        self.t_sym = sp.symbols('t', real=True)
        self.z_sym = sp.symbols('z')
        self.d_func = sp.Function('delta_0')
        
    def parse_input(self, delta_specs_data, equation_str):
        for spec in delta_specs_data:
            self.delta_specs.append(DeltaSpec(spec['time_expr'], spec['active_at']))
            
        import re
        local_dict = {'t': self.t_sym, 'd': self.d_func, 'D': self.d_func}
                
        # Preprocess equation
        eq_clean = equation_str.split('=')[-1].strip() # Take right side if "y(t) =" is present
        eq_clean = eq_clean.replace('D(', 'd(') # Standardize D to d
        
        # Auto-detect symbols to prevent implicit multiplication from splitting 'c2' into 'c*2'
        for var_match in re.finditer(r'[a-zA-Z_]\w*', eq_clean):
            var_name = var_match.group()
            if var_name not in local_dict:
                local_dict[var_name] = sp.symbols(var_name)
        
        self.equation_sym = parse_expr(eq_clean, local_dict=local_dict, transformations=transformations)
        
        terms = self.equation_sym.as_ordered_terms() if isinstance(self.equation_sym, sp.Add) else [self.equation_sym]
        for term in terms:
            d_funcs = term.atoms(sp.Function)
            if not d_funcs:
                raise ValueError(f"Il termine '{term}' nell'equazione non contiene alcuna funzione delta. Assicurati che ogni costante moltiplichi una funzione d(...).")
            
            d_term_candidates = [f for f in d_funcs if f.func == self.d_func]
            if not d_term_candidates:
                 raise ValueError(f"Il termine '{term}' contiene una funzione non riconosciuta. Usa d(...) o D(...) per le delta.")
            if len(d_term_candidates) > 1:
                raise ValueError(f"Il termine '{term}' contiene più funzioni delta moltiplicate tra loro, formato non supportato.")
                
            d_term = d_term_candidates[0]
            constant = sp.simplify(term / d_term)
            
            # find matching spec
            matched = False
            for spec in self.delta_specs:
                if sp.simplify(d_term.args[0] - spec.time_expr_sym) == 0:
                    spec.constant = constant
                    spec.found = True
                    matched = True
                    break
                    
            if not matched:
                raise ValueError(f"La funzione {d_term} usata nell'equazione non è stata definita nelle specifiche delle Delta. Aggiungila nelle specifiche oppure rimuovila dall'equazione.")
                
        # Verifica che tutte le delta definite abbiano un moltiplicatore
        for spec in self.delta_specs:
            if not spec.found:
                raise ValueError(f"La funzione delta d({spec.time_expr_str}) definita nelle specifiche non compare nell'equazione. Rimuovila dalle specifiche o aggiungila all'equazione.")
        
        # Step LaTeX
        specs_latex = []
        for spec in self.delta_specs:
            specs_latex.append(f"<div style='margin-bottom: 5px;'>\\[ {to_latex(self.d_func(spec.time_expr_sym))} \\rightarrow 1 \\text{{ se }} t={spec.active_at} \\implies \\text{{Costante: }} {to_latex(spec.constant)} \\]</div>")
        
        self.steps_latex.append({
            "title": "Parsing dell'Input ed Estrazione Costanti",
            "content": f"<div style='margin-bottom: 10px;'><strong>Specifiche Delta:</strong></div>" + "".join(specs_latex) + f"<div style='margin-top: 15px; margin-bottom: 10px;'><strong>Equazione Inserita:</strong></div><div>\\[ y(t) = {to_latex(self.equation_sym)} \\]</div>",
            "is_html": True
        })

    def compute_impulse_response(self):
        if not self.delta_specs:
            self.M = 0
        else:
            self.M = max(spec.active_at for spec in self.delta_specs)
            
        h_calc_html = ""
        for t_val in range(self.M + 1):
            h_val = 0
            active_terms_str = []
            explanations = []
            for spec in self.delta_specs:
                if t_val == spec.active_at:
                    h_val += spec.constant
                    active_terms_str.append(f"\\left({to_latex(spec.constant)}\\right) \\cdot 1")
                    explanations.append(f"\\( {to_latex(self.d_func(spec.time_expr_sym))} \\) è attiva (vale 1)")
                else:
                    explanations.append(f"\\( {to_latex(self.d_func(spec.time_expr_sym))} \\) è inattiva (vale 0)")
            
            self.h_values[t_val] = sp.simplify(h_val)
            
            h_calc_html += f"<div style='margin-top: 15px;'><strong>Per \\( t={t_val} \\):</strong><ul>"
            for expl in explanations:
                h_calc_html += f"<li>{expl}</li>"
            h_calc_html += "</ul></div>"
            
            if active_terms_str:
                h_calc_html += f"<div>\\[ h({t_val}) = " + " + ".join(active_terms_str) + f" = {to_latex(self.h_values[t_val])} \\]</div>"
            else:
                h_calc_html += f"<div>\\[ h({t_val}) = 0 \\]</div>"
            
        self.steps_latex.append({
            "title": "Calcolo della Risposta Impulsiva \\( h(t) \\)",
            "content": h_calc_html,
            "is_html": True
        })

    def compute_transfer_function(self):
        N_terms = []
        for t_val in range(self.M + 1):
            N_terms.append(self.h_values[t_val] * self.z_sym**(self.M - t_val))
            
        self.N_z = sum(N_terms)
        self.D_z = self.z_sym**self.M
        self.H_z = self.N_z / self.D_z
        
        sum_str = " + ".join([f"h({t}) z^{{{-t}}}" for t in range(self.M + 1)])
        subst_str = " + ".join([f"\\left({to_latex(self.h_values[t])}\\right) z^{{{-t}}}" for t in range(self.M + 1)])
        
        content_html = f"<div>\\[ H(z) = \\sum_{{t=0}}^{{{self.M}}} h(t) z^{{-t}} \\]</div>"
        content_html += f"<div>\\[ H(z) = {sum_str} \\]</div>"
        content_html += f"<div>\\[ H(z) = {subst_str} \\]</div>"
        content_html += f"<div>\\[ H(z) = \\frac{{{to_latex(sp.expand(self.N_z))}}}{{{to_latex(self.D_z)}}} \\]</div>"
        
        self.steps_latex.append({
            "title": "Funzione di Trasferimento \\( H(z) \\)",
            "content": content_html,
            "is_html": True
        })

    def identify_coefficients(self):
        # Step 4: Order
        self.n = self.M
        self.steps_latex.append({
            "title": "Determinazione dell'Ordine",
            "content": f"\\text{{Grado del denominatore }} D(z) = z^{{{self.M}}} \\implies n = {self.n}"
        })
        
        # Step 5: Direct term D
        N_poly = sp.expand(self.N_z)
        D_poly = sp.expand(self.D_z)
        
        grado_num = sp.degree(N_poly, gen=self.z_sym)
        grado_den = self.n
        
        if grado_num < grado_den:
            self.D_val = sp.sympify(0)
            N_din = N_poly
            case_desc = "Grado numeratore < Grado denominatore \\implies D = 0"
        elif grado_num == grado_den:
            self.D_val = N_poly.coeff(self.z_sym, self.n) / D_poly.coeff(self.z_sym, self.n)
            N_din = sp.expand(N_poly - self.D_val * D_poly)
            case_desc = f"Grado numeratore = Grado denominatore \\implies D = {to_latex(self.D_val)}"
        else:
            raise ValueError("Sistema non causalmente realizzabile (grado numeratore > grado denominatore)")
            
        self.steps_latex.append({
            "title": "Identificazione del Termine Diretto \\( D \\)",
            "content": case_desc
        })
        
        # Step 6: Coefficients
        self.a_coeffs = [D_poly.coeff(self.z_sym, i) for i in range(self.n)]
        self.b_coeffs = [N_din.coeff(self.z_sym, i) for i in range(self.n)]
        
        coeff_html = "<div style='display: flex; justify-content: space-around; margin-top: 15px;'>"
        
        # a_i coeffs
        coeff_html += "<div><strong>Coefficienti Denominatore:</strong><ul>"
        for i in range(self.n):
            coeff_html += f"<li>\\( a_{{{i}}} = {to_latex(self.a_coeffs[i])} \\)</li>"
        coeff_html += "</ul></div>"
        
        # b_i coeffs
        coeff_html += "<div><strong>Coefficienti Numeratore (dinamico):</strong><ul>"
        for i in range(self.n):
            coeff_html += f"<li>\\( b_{{{i}}} = {to_latex(self.b_coeffs[i])} \\)</li>"
        coeff_html += "</ul></div>"
        
        coeff_html += "</div>"
        coeff_html += f"<div style='text-align: center; margin-top: 10px;'><strong>Termine diretto:</strong> \\( D = {to_latex(self.D_val)} \\)</div>"
        
        if self.n > 0:
            self.steps_latex.append({
                "title": "Identificazione dei Coefficienti",
                "content": coeff_html,
                "is_html": True
            })
        else:
             self.steps_latex.append({
                "title": "Identificazione dei Coefficienti",
                "content": "<p>Sistema di ordine 0, nessun coefficiente dinamico. Solo termine diretto \\( D \\).</p>",
                "is_html": True
            })

    def build_companion_matrices(self):
        if self.n == 0:
            self.A = sp.zeros(0, 0)
            self.B = sp.zeros(0, 1)
            self.C = sp.zeros(1, 0)
            self.D_mat = sp.Matrix([[self.D_val]])
        elif self.n == 1:
            self.A = sp.Matrix([[-self.a_coeffs[0]]])
            self.B = sp.Matrix([[1]])
            self.C = sp.Matrix([[self.b_coeffs[0]]])
            self.D_mat = sp.Matrix([[self.D_val]])
        else:
            self.A = sp.zeros(self.n, self.n)
            for i in range(self.n - 1):
                self.A[i, i + 1] = 1
            for i in range(self.n):
                self.A[self.n - 1, i] = -self.a_coeffs[i]
                
            self.B = sp.zeros(self.n, 1)
            self.B[self.n - 1, 0] = 1
            
            self.C = sp.zeros(1, self.n)
            for i in range(self.n):
                self.C[0, i] = self.b_coeffs[i]
                
            self.D_mat = sp.Matrix([[self.D_val]])
            
        html_content = ""
        if self.n > 0:
            html_content = f"""
            <div style="display: flex; flex-direction: column; gap: 20px; align-items: center; margin-top: 15px; width: 100%;">
                <div style="display: flex; gap: 50px; justify-content: center; align-items: center; width: 100%;">
                    <div style="flex: 1; text-align: right;">\\[ \\text{{Matrice dinamica }} A \\; ({self.n} \\times {self.n}): \\quad A = {to_latex(self.A)} \\]</div>
                    <div style="flex: 1; text-align: left;">\\[ \\text{{Matrice ingresso }} B \\; ({self.n} \\times 1): \\quad B = {to_latex(self.B)} \\]</div>
                </div>
                <div style="display: flex; gap: 50px; justify-content: center; align-items: center; width: 100%;">
                    <div style="flex: 1; text-align: right;">\\[ \\text{{Matrice uscita }} C \\; (1 \\times {self.n}): \\quad C = {to_latex(self.C)} \\]</div>
                    <div style="flex: 1; text-align: left;">\\[ \\text{{Legame diretto }} D \\; (1 \\times 1): \\quad D = {to_latex(self.D_mat)} \\]</div>
                </div>
            </div>
            """
        else:
            html_content = f"""
            <div style="display: flex; flex-direction: column; gap: 20px; align-items: center; margin-top: 15px; width: 100%;">
                <div style="display: flex; gap: 50px; justify-content: center; align-items: center;">
                    <div>\\[ \\text{{Legame diretto }} D \\; (1 \\times 1): \\quad D = {to_latex(self.D_mat)} \\]</div>
                </div>
            </div>
            """
            
        self.steps_latex.append({
            "title": "Costruzione Matrici in Forma Compagna",
            "content": html_content,
            "is_html": True
        })

    def verify_solution(self):
        if self.n > 0:
            I = sp.eye(self.n)
            zI_minus_A = self.z_sym * I - self.A
            zI_minus_A_inv = zI_minus_A.inv()
            H_z_ver = self.C * zI_minus_A_inv * self.B + self.D_mat
            H_z_ver_simplified = sp.simplify(H_z_ver[0, 0])
            
            # Formattiamo H_z originale raccogliendo a denominatore comune per verifica visuale
            H_z_orig = sp.cancel(self.H_z)
            
            content_html = f"<div>\\[ H_{{ver}}(z) = {to_latex(H_z_ver_simplified)} \\]</div>"
            content_html += f"<div>\\[ H_{{orig}}(z) = {to_latex(H_z_orig)} \\]</div>"
            
            diff = sp.simplify(H_z_ver_simplified - H_z_orig)
            if diff == 0:
                content_html += "<div style='text-align: center; margin-top: 15px; padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px; border: 1px solid #c3e6cb;'>"
                content_html += "<strong>Le due funzioni di trasferimento coincidono perfettamente, la soluzione è verificata e corretta!</strong></div>"
            else:
                content_html += "<div style='text-align: center; margin-top: 15px; padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px; border: 1px solid #f5c6cb;'>"
                content_html += "<strong>Le due funzioni di trasferimento NON coincidono. C'è un problema nella costruzione.</strong></div>"
            
            self.steps_latex.append({
                "title": "Verifica \\( H(z) = C(zI - A)^{-1}B + D \\)",
                "content": content_html,
                "is_html": True
            })
            
    def build_state_equations(self):
        if self.n > 0:
            x_k1 = sp.Matrix([sp.symbols(f'x_{{{i+1}}}(k+1)') for i in range(self.n)])
            x_k = sp.Matrix([sp.symbols(f'x_{{{i+1}}}(k)') for i in range(self.n)])
            u_k = sp.symbols('u(k)')
            y_k = sp.symbols('y(k)')
            
            content_html = f"<div style='margin-top: 15px;'>\\[ {to_latex(x_k1)} = {to_latex(self.A)} {to_latex(x_k)} + {to_latex(self.B)} {to_latex(u_k)} \\]</div>"
            content_html += f"<div style='margin-top: 15px;'>\\[ {to_latex(y_k)} = {to_latex(self.C)} {to_latex(x_k)} + {to_latex(self.D_mat)} {to_latex(u_k)} \\]</div>"
            
            self.steps_latex.append({
                "title": "Equazioni di Stato Finali (Tempo Discreto)",
                "content": content_html,
                "is_html": True
            })

    def solve(self, delta_specs_data, equation_str):
        self.parse_input(delta_specs_data, equation_str)
        self.compute_impulse_response()
        self.compute_transfer_function()
        self.identify_coefficients()
        self.build_companion_matrices()
        self.build_state_equations()
        self.verify_solution()
        return self.steps_latex

# ---- ROUTES ----

@delta_da_problema_a_soluzione_bp.route('/delta_da_problema_a_soluzione.html')
def pagina_html():
    # Rotta GET per servire la pagina
    return render_template('delta_da_problema_a_soluzione.html')

@delta_da_problema_a_soluzione_bp.route('/api/delta_da_problema_a_soluzione', methods=['POST'])
def api_risolvi():
    data = request.get_json()
    delta_specs = data.get("delta_specs", [])
    equation_str = data.get("equation", "")
    
    if not delta_specs or not equation_str:
        return jsonify({"success": False, "error": "Parametri mancanti: inserisci almeno una delta e l'equazione."})
        
    try:
        solver = ImpulseResponseSolver()
        steps = solver.solve(delta_specs, equation_str)
        return jsonify({"success": True, "latex": steps})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
