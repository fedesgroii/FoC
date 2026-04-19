from flask import Blueprint, request, jsonify, render_template
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from .utils import transformations_basic as transformations
import traceback
import re

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

class ShiftSolver:
    def __init__(self):
        self.delta_specs = []
        self.equation_sym = None
        self.n = 0
        self.steps_latex = []
        self.t_sym = sp.symbols('t', real=True)
        self.d_func = sp.Function('delta_0')
        self.A = None
        self.B = None
        self.C = None
        self.D_mat = None
        self.diff_eq_html = ""
        self.state_vars_html = ""
        self.matrices_html = ""
        
    def parse_input(self, delta_specs_data, equation_str):
        for spec in delta_specs_data:
            self.delta_specs.append(DeltaSpec(spec['time_expr'], spec['active_at']))
            
        local_dict = {'t': self.t_sym, 'd': self.d_func, 'D': self.d_func}
                
        eq_clean = equation_str.split('=')[-1].strip()
        eq_clean = eq_clean.replace('D(', 'd(')
        
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
                raise ValueError(f"Il termine '{term}' contiene più funzioni delta moltiplicate tra loro.")
                
            d_term = d_term_candidates[0]
            constant = sp.simplify(term / d_term)
            
            matched = False
            for spec in self.delta_specs:
                if sp.simplify(d_term.args[0] - spec.time_expr_sym) == 0:
                    spec.constant = constant
                    spec.found = True
                    matched = True
                    break
                    
            if not matched:
                raise ValueError(f"La funzione {d_term} usata nell'equazione non è stata definita nelle specifiche delle Delta.")
                
        for spec in self.delta_specs:
            if not spec.found:
                raise ValueError(f"La funzione delta d({spec.time_expr_str}) definita nelle specifiche non compare nell'equazione.")
        
        self.n = len(self.delta_specs)
        
        specs_latex = []
        for spec in self.delta_specs:
            specs_latex.append(f"<div style='margin-bottom: 5px;'>\\[ {to_latex(self.d_func(spec.time_expr_sym))} \\rightarrow 1 \\text{{ se }} t={spec.active_at} \\implies \\text{{Costante: }} {to_latex(spec.constant)} \\]</div>")
        
        self.steps_latex.append({
            "title": "Parsing dell'Input ed Estrazione Costanti",
            "content": f"<div style='margin-bottom: 10px;'><strong>Specifiche Delta identificate (Ordine \\( n={self.n} \\)):</strong></div>" + "".join(specs_latex) + f"<div style='margin-top: 15px; margin-bottom: 10px;'><strong>Equazione Inserita:</strong></div><div>\\[ y(t) = {to_latex(self.equation_sym)} \\]</div>",
            "is_html": True
        })

    def compute_shifts(self):
        content_html = "<p>Calcoliamo gli shift temporali dell'equazione sostituendo \\( t \\) con \\( t+k \\):</p>"
        
        for k in range(self.n + 1):
            if k == 0:
                shifted_expr = self.equation_sym
            else:
                shifted_expr = self.equation_sym.subs(self.t_sym, self.t_sym + k)
                
            # Expand and format
            content_html += f"<div style='margin-top: 10px;'><strong>Shift \\( k={k} \\):</strong> \\[ y(t+{k}) = {to_latex(shifted_expr)} \\]</div>"
            
        content_html += "<p style='margin-top: 15px;'>Notiamo che per tempi sufficientemente lunghi, gli argomenti delle delta diventano sempre maggiori di zero e le funzioni si annullano, portando a un'equazione omogenea per \\( y(t+n) \\).</p>"
            
        self.steps_latex.append({
            "title": "Calcolo degli Shift Temporali",
            "content": content_html,
            "is_html": True
        })

    def compute_difference_equation(self):
        self.diff_eq_html = f"\\[ y(t+{self.n}) = 0 \\]"
        
        self.steps_latex.append({
            "title": "Derivazione Equazione alle Differenze",
            "content": f"<p>Poiché tutte le funzioni Delta hanno supporto finito, si spengono progressivamente. Dopo \\( {self.n} \\) istanti, l'uscita libera del sistema (senza ingressi futuri) diventa zero.</p><div>{self.diff_eq_html}</div>",
            "is_html": True
        })

    def build_state_space(self):
        if self.n == 0:
            return
            
        state_vars_str = "<ul>"
        for i in range(1, self.n + 1):
            if i == 1:
                state_vars_str += f"<li>\\( x_{i}(t) = y(t) \\)</li>"
            else:
                state_vars_str += f"<li>\\( x_{i}(t) = y(t+{i-1}) \\)</li>"
        state_vars_str += "</ul>"
        self.state_vars_html = state_vars_str
        
        self.steps_latex.append({
            "title": "Scelta delle Variabili di Stato",
            "content": f"<p>Definiamo le variabili di stato canoniche basate sugli shift dell'uscita:</p>{self.state_vars_html}",
            "is_html": True
        })
        
        self.A = sp.zeros(self.n, self.n)
        for i in range(self.n - 1):
            self.A[i, i + 1] = 1
            
        self.B = sp.zeros(self.n, 1)
        self.C = sp.zeros(1, self.n)
        self.C[0, 0] = 1
        self.D_mat = sp.Matrix([[0]])
        
        html_content = f"""
        <div style="display: flex; flex-direction: column; gap: 20px; align-items: center; margin-top: 15px; width: 100%;">
            <p>Il sistema in forma autonoma (risposta all'impulso vista come evoluzione libera da stato iniziale) presenta una matrice dinamica <strong>nilpotente</strong> (spostamento puro).</p>
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
        self.matrices_html = html_content
        
        self.steps_latex.append({
            "title": "Costruzione Matrici dello Spazio di Stato",
            "content": html_content,
            "is_html": True
        })

    def solve(self, delta_specs_data, equation_str):
        self.parse_input(delta_specs_data, equation_str)
        self.compute_shifts()
        self.compute_difference_equation()
        self.build_state_space()
        
        return {
            "steps": self.steps_latex,
            "difference_equation": self.diff_eq_html,
            "state_variables": self.state_vars_html,
            "matrices": self.matrices_html
        }

# ---- ROUTES ----

@delta_da_problema_a_soluzione_bp.route('/delta_da_problema_a_soluzione.html')
def pagina_html():
    return render_template('delta_da_problema_a_soluzione.html')

@delta_da_problema_a_soluzione_bp.route('/api/delta_da_problema_a_soluzione', methods=['POST'])
def api_risolvi():
    data = request.get_json()
    delta_specs = data.get("delta_specs", [])
    equation_str = data.get("equation", "")
    
    if not delta_specs or not equation_str:
        return jsonify({"success": False, "error": "Parametri mancanti: inserisci almeno una delta e l'equazione."})
        
    try:
        solver = ShiftSolver()
        result = solver.solve(delta_specs, equation_str)
        return jsonify({
            "success": True, 
            "data": result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
