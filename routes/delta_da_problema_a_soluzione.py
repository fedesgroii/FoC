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
        self.d_func = sp.Function('d')
        
    def parse_input(self, delta_specs_data, constants_str, equation_str):
        for spec in delta_specs_data:
            self.delta_specs.append(DeltaSpec(spec['time_expr'], spec['active_at']))
            
        # Parse constants
        const_names = [c.strip() for c in constants_str.split(',')]
        local_dict = {'t': self.t_sym, 'd': self.d_func, 'D': self.d_func}
        for c in const_names:
            if c:
                local_dict[c] = sp.symbols(c)
                self.constants.append(c)
                
        # Preprocess equation
        eq_clean = equation_str.split('=')[-1].strip() # Take right side if "y(t) =" is present
        eq_clean = eq_clean.replace('D(', 'd(') # Standardize D to d
        
        self.equation_sym = parse_expr(eq_clean, local_dict=local_dict, transformations=transformations)
        
        # Step LaTeX
        specs_latex = []
        for spec in self.delta_specs:
            specs_latex.append(f"\\delta({to_latex(spec.time_expr_sym)}) \\rightarrow 1 \\text{{ se }} t={spec.active_at}")
        
        self.steps_latex.append({
            "title": "Parsing dell'Input",
            "content": f"\\begin{{aligned}} &\\text{{Specifiche Delta:}}\\\\ &" + " \\\\ &".join(specs_latex) + f" \\\\ \\\\ &\\text{{Equazione:}} \\\\ &y(t) = {to_latex(self.equation_sym)} \\end{{aligned}}"
        })

    def compute_impulse_response(self):
        if not self.delta_specs:
            self.M = 0
        else:
            self.M = max(spec.active_at for spec in self.delta_specs)
            
        h_calc_latex = []
        for t_val in range(self.M + 1):
            eq_t = self.equation_sym
            for spec in self.delta_specs:
                is_active = (t_val == spec.active_at)
                term = self.d_func(spec.time_expr_sym)
                eq_t = eq_t.subs(term, 1 if is_active else 0)
                
            self.h_values[t_val] = sp.simplify(eq_t)
            h_calc_latex.append(f"h({t_val}) = {to_latex(self.h_values[t_val])}")
            
        self.steps_latex.append({
            "title": "Calcolo della Risposta Impulsiva \\( h(t) \\)",
            "content": f"\\begin{{aligned}} " + " \\\\ ".join(h_calc_latex) + " \\end{{aligned}}"
        })

    def compute_transfer_function(self):
        N_terms = []
        for t_val in range(self.M + 1):
            N_terms.append(self.h_values[t_val] * self.z_sym**(self.M - t_val))
            
        self.N_z = sum(N_terms)
        self.D_z = self.z_sym**self.M
        self.H_z = self.N_z / self.D_z
        
        self.steps_latex.append({
            "title": "Funzione di Trasferimento \\( H(z) \\)",
            "content": f"H(z) = \\sum_{{t=0}}^{{{self.M}}} h(t) z^{{-t}} = \\frac{{{to_latex(sp.expand(self.N_z))}}}{{{to_latex(self.D_z)}}}"
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
        
        coeff_latex = []
        for i in range(self.n):
            coeff_latex.append(f"a_{{{i}}} = {to_latex(self.a_coeffs[i])}, \\quad b_{{{i}}} = {to_latex(self.b_coeffs[i])}")
            
        if self.n > 0:
            self.steps_latex.append({
                "title": "Identificazione dei Coefficienti",
                "content": f"\\begin{{aligned}} " + " \\\\ ".join(coeff_latex) + " \\end{{aligned}}"
            })
        else:
             self.steps_latex.append({
                "title": "Identificazione dei Coefficienti",
                "content": "\\text{Sistema di ordine 0, nessun coefficiente dinamico.}"
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
            <div style="display: flex; flex-direction: column; gap: 20px; align-items: center; margin-top: 15px;">
                <div style="display: flex; gap: 50px; justify-content: center; align-items: center;">
                    <div>\\[ A = {to_latex(self.A)} \\]</div>
                    <div>\\[ B = {to_latex(self.B)} \\]</div>
                </div>
                <div style="display: flex; gap: 50px; justify-content: center; align-items: center;">
                    <div>\\[ C = {to_latex(self.C)} \\]</div>
                    <div>\\[ D = {to_latex(self.D_mat)} \\]</div>
                </div>
            </div>
            """
        else:
            html_content = f"""
            <div style="display: flex; flex-direction: column; gap: 20px; align-items: center; margin-top: 15px;">
                <div style="display: flex; gap: 50px; justify-content: center; align-items: center;">
                    <div>\\[ D = {to_latex(self.D_mat)} \\]</div>
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
            
            self.steps_latex.append({
                "title": "Verifica \\( H(z) = C(zI - A)^{-1}B + D \\)",
                "content": f"H_{{ver}}(z) = {to_latex(H_z_ver_simplified)} \\quad \\text{{vs}} \\quad H_{{orig}}(z) = {to_latex(H_z_orig)}"
            })

    def solve(self, delta_specs_data, constants_str, equation_str):
        self.parse_input(delta_specs_data, constants_str, equation_str)
        self.compute_impulse_response()
        self.compute_transfer_function()
        self.identify_coefficients()
        self.build_companion_matrices()
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
    constants_str = data.get("constants", "")
    equation_str = data.get("equation", "")
    
    if not delta_specs or not equation_str:
        return jsonify({"success": False, "error": "Parametri mancanti: inserisci almeno una delta e l'equazione."})
        
    try:
        solver = ImpulseResponseSolver()
        steps = solver.solve(delta_specs, constants_str, equation_str)
        return jsonify({"success": True, "latex": steps})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
