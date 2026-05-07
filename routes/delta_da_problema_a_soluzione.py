from flask import Blueprint, request, jsonify, render_template
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from .utils import transformations_basic as transformations
import traceback
import re

delta_da_problema_a_soluzione_bp = Blueprint("delta_da_problema_a_soluzione", __name__)

def convert_power_operator(expr_str):
    """
    Converte l'operatore ^ in ** per le espressioni matematiche.
    Gestisce casi come: 5^t, 5^(t), (t+1)^2, d(t^2), ecc.
    """
    if not expr_str:
        return expr_str
    # Sostituzione diretta di ^ con **. In questo contesto matematico,
    # ^ è sempre inteso come elevamento a potenza, non XOR bit-wise.
    return expr_str.replace('^', '**')


def to_latex(expr):
    return sp.latex(expr)

class DeltaSpec:
    def __init__(self, time_expr_str, active_at):
        # Convertiamo ^ in ** prima del parsing
        self.time_expr_str = convert_power_operator(time_expr_str)

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
        self.input_expr_sym = None
        self.has_external_input = False
        self.impulse_response = None
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
        self.free_terms_sym = sp.Integer(0)
        
    def parse_input(self, delta_specs_data, equation_str, input_expr_str=None):
        for spec in delta_specs_data:
            self.delta_specs.append(DeltaSpec(spec['time_expr'], spec['active_at']))
            
        local_dict = {'t': self.t_sym, 'd': self.d_func, 'D': self.d_func}
                
        eq_clean = equation_str.split('=')[-1].strip()
        eq_clean = eq_clean.replace('D(', 'd(')
        # Convertiamo ^ in ** prima del parsing
        eq_clean = convert_power_operator(eq_clean)

        
        for var_match in re.finditer(r'[a-zA-Z_]\w*', eq_clean):
            var_name = var_match.group()
            if var_name not in local_dict:
                local_dict[var_name] = sp.symbols(var_name)
        
        self.equation_sym = parse_expr(eq_clean, local_dict=local_dict, transformations=transformations)

        if input_expr_str:
            input_clean = convert_power_operator(input_expr_str.strip())
            # Aggiungiamo eventuali nuove variabili simboliche dall'input
            for var_match in re.finditer(r'[a-zA-Z_]\w*', input_clean):
                var_name = var_match.group()
                if var_name not in local_dict:
                    local_dict[var_name] = sp.symbols(var_name)
            
            self.input_expr_sym = parse_expr(input_clean, local_dict=local_dict, transformations=transformations)
            self.has_external_input = True
            
            # Verifichiamo che l'ingresso compaia nell'equazione
            # Espandiamo entrambi per un confronto più robusto
            eq_expanded = sp.expand(self.equation_sym)
            input_expanded = sp.expand(self.input_expr_sym)
            
            # Sottraiamo l'ingresso dall'equazione
            self.equation_without_input = sp.simplify(self.equation_sym - self.input_expr_sym)
            
            # Se la sottrazione non ha cambiato nulla (l'ingresso non era presente), solleva errore
            if sp.simplify(self.equation_without_input - self.equation_sym) == 0 and self.input_expr_sym != 0:
                raise ValueError(f"L'ingresso specificato '{input_expr_str}' non è presente nell'equazione.")
        else:
            self.equation_without_input = self.equation_sym
        
        # Extract terms and separate deltas from free terms
        # Il parsing delle delta deve avvenire sull'equazione originale (con ingresso)
        terms = self.equation_sym.as_ordered_terms() if isinstance(self.equation_sym, sp.Add) else [self.equation_sym]
        delta_terms = []
        self.free_terms_sym = sp.Integer(0)
        
        for term in terms:
            d_funcs = [f for f in term.atoms(sp.Function) if f.func == self.d_func]
            if d_funcs:
                delta_terms.append(term)
                
                # Validation of the delta term
                d_term = d_funcs[0]
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
            else:
                self.free_terms_sym += term

        # Se è presente un ingresso esterno, lo sottraiamo dai termini liberi
        if self.has_external_input:
            self.free_terms_sym = sp.simplify(self.free_terms_sym - self.input_expr_sym)

        for spec in self.delta_specs:
            if not spec.found:
                raise ValueError(f"La funzione delta d({spec.time_expr_str}) definita nelle specifiche non compare nell'equazione.")
        
        # Determine recurrence polynomial from free terms
        z = sp.symbols('z')
        poles = set()
        free_expanded = sp.expand(self.free_terms_sym)

        
        # Simple pole extraction (base**t, exp(c*t), constants)
        for p in free_expanded.atoms(sp.Pow):
            if p.exp.has(self.t_sym) and not p.base.has(self.t_sym):
                coeff = p.exp.coeff(self.t_sym)
                if coeff is not None:
                    poles.add(p.base ** coeff)
        for e in free_expanded.atoms(sp.exp):
            c = e.args[0].coeff(self.t_sym)
            if c is not None:
                poles.add(sp.exp(c))
        
        if free_expanded != 0:
            if not poles: # Poly in t or just constants
                poles.add(sp.Integer(1))
            # Handle multiplicities for polynomials (t, t**2 etc -> pole at 1)
            # Calcolo del grado polinomiale di t in modo sicuro
            try:
                deg_t = sp.degree(free_expanded, self.t_sym)
            except:
                # Se non è un polinomio puro (es. 5**t), il grado polinomiale è 0
                deg_t = 0
                
            if deg_t > 0:
                # Se c'è una parte polinomiale, assicuriamoci che il polo 1 sia presente
                poles.add(sp.Integer(1))
                # Per ora usiamo una gestione semplice della molteplicità:
                # se il grado è > 0, il polo 1 è già aggiunto.
                pass 

        p_modal = sp.Integer(1)
        if not poles:
            p_modal = z # Nilpotent
        else:
            for p in poles:
                p_modal *= (z - p)
        
        self.p_modal_poly = sp.Poly(p_modal, z)
        deg_p = self.p_modal_poly.degree()
        
        # Max delay from deltas
        max_delay = -1
        for spec in self.delta_specs:
            # Try to find constant k in t-k
            # active_at is already k
            if spec.active_at > max_delay:
                max_delay = spec.active_at
        
        # Total order N
        self.n = max(max_delay + 1, deg_p)
        
        specs_latex = []
        for spec in self.delta_specs:
            specs_latex.append(f"<div style='margin-bottom: 5px;'>\\[ {to_latex(self.d_func(spec.time_expr_sym))} \\rightarrow 1 \\text{{ se }} t={spec.active_at} \\implies \\text{{Costante: }} {to_latex(spec.constant)} \\]</div>")
        
        free_part_latex = f" + {to_latex(self.free_terms_sym)}" if self.free_terms_sym != 0 else ""
        
        input_info_html = ""
        if self.has_external_input:
            input_info_html = f"<div style='margin-top: 15px;'><strong>Ingresso Esterno:</strong> \\[ u(t) = {to_latex(self.input_expr_sym)} \\]</div>"

        self.steps_latex.append({
            "title": "Parsing dell'Input ed Estrazione Costanti",
            "content": f"<div style='margin-bottom: 10px;'><strong>Specifiche Delta identificate:</strong></div>" + "".join(specs_latex) + 
                       input_info_html +
                       f"<div style='margin-top: 15px;'><strong>Termine Forzante del Sistema (senza ingresso):</strong> \\[ f(t) = {to_latex(self.free_terms_sym)} \\]</div>" +
                       f"<div style='margin-top: 15px; margin-bottom: 10px;'><strong>Equazione Completa Inserita:</strong></div><div>\\[ y(t) = {to_latex(self.equation_sym)} \\]</div>" +
                       f"<div style='margin-top: 10px;'><strong>Ordine del Sistema identificato:</strong> \\( n = {self.n} \\)</div>",
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
            
        content_html += "<p style='margin-top: 15px;'>Notiamo che per tempi sufficientemente lunghi, gli impulsi si esauriscono e l'uscita segue la dinamica del termine libero, soddisfacendo l'equazione caratteristica.</p>"

            
        self.steps_latex.append({
            "title": "Calcolo degli Shift Temporali",
            "content": content_html,
            "is_html": True
        })

    def get_y_val(self, t_val, expr=None):
        if expr is None:
            expr = self.equation_sym
            
        subs_dict = {self.t_sym: t_val}
        # Handle d(t-k)
        for f in expr.atoms(sp.Function):
            if f.func == self.d_func:
                arg_val = f.args[0].subs(self.t_sym, t_val)
                # Confronto simbolico/numerico robusto per l'argomento della delta
                diff = sp.simplify(arg_val)
                subs_dict[f] = 1 if diff == 0 else 0
        
        # Sostituiamo t e valutiamo
        return expr.subs(subs_dict).subs(self.t_sym, t_val)

    def compute_impulse_response(self):
        """
        Calcola la risposta all'impulso h(t) = y(t) quando u(t)=delta_0(t) 
        e condizioni iniziali nulle.
        In questo contesto, h(t) è dato dai valori di free_terms_sym (la parte forzata pura).
        """
        n = self.n
        h_vals = []
        for k in range(n + 1):
            val = self.get_y_val(k, expr=self.free_terms_sym)
            h_vals.append(val)
        self.impulse_response = h_vals
        return h_vals


    def compute_difference_equation(self):
        z = sp.symbols('z')
        coeffs = self.p_modal_poly.all_coeffs()
        # The recurrence is sum(coeffs[i] * y(t + (deg - i))) = 0
        deg = self.p_modal_poly.degree()
        
        lhs_terms = []
        for i, c in enumerate(coeffs):
            shift = deg - i
            term = f"{to_latex(c)} y(t+{shift})" if c != 1 else f"y(t+{shift})"
            if c != 0:
                lhs_terms.append(term)
        
        # For the display, we shift the recurrence to the system order n
        final_shift = self.n - deg
        final_eq_parts = []
        for i, c in enumerate(coeffs):
            shift = deg - i + final_shift
            term = f"{to_latex(c)} y(t+{shift})" if c != 1 else f"y(t+{shift})"
            if c != 0:
                final_eq_parts.append(term)
        
        final_eq_latex = " + ".join(final_eq_parts).replace("+ -", "- ") + " = 0"
        
        self.diff_eq_html = f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #f0f4f8 100%);
            border: 3px solid #007bff;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <h3 style="margin-top: 0; color: #007bff; font-size: 1.3rem; text-transform: uppercase; letter-spacing: 1px;">Equazione alle Differenze Finale</h3>
            <p style="font-size: 0.9rem; color: #666; margin-bottom: 15px;">(Valida per \( t \geq {max(0, self.n - deg)} \))</p>
            <div style="font-size: 1.8rem; font-weight: bold; color: #212529; overflow-x: auto; padding: 10px 0;">
                \\( {final_eq_latex} \\)
            </div>
        </div>
        """
        
        self.steps_latex.append({
            "title": "Derivazione Equazione alle Differenze",
            "content": f"<p>L'equazione caratteristica è determinata dai poli del termine libero: \\( P(z) = {to_latex(self.p_modal_poly.as_expr())} \\).</p>{self.diff_eq_html}",
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
            
        # Last row of A comes from the recurrence
        deg = self.p_modal_poly.degree()
        coeffs = self.p_modal_poly.all_coeffs() # [1, p_{D-1}, ..., p_0]
        for i in range(1, deg + 1):
            self.A[self.n - 1, self.n - i] = -coeffs[i]
            
        # Determiniamo B e D
        if self.has_external_input:
            # CASO B: Con ingresso esterno. Usiamo la risposta all'impulso h(t)
            h_vals = self.compute_impulse_response()
            self.B = sp.Matrix(h_vals[1:])
            self.D_mat = sp.Matrix([[h_vals[0]]])
            explanation_bd = f"Poiché è presente un ingresso esterno \\( u(t) \\), le matrici \\( B \\) e \\( D \\) sono determinate dalla risposta all'impulso del sistema \\( h(t) \\)."
            impulse_res_latex = f"\\( h(t) = [{', '.join([to_latex(v) for v in h_vals])}, \\dots] \\)"
        else:
            # CASO A: Nessun ingresso esterno. B e D dai valori di free_terms_sym (vecchio comportamento)
            y_free_vals = [self.get_y_val(k, expr=self.free_terms_sym) for k in range(self.n + 1)]
            self.B = sp.Matrix(y_free_vals[1:])
            self.D_mat = sp.Matrix([[y_free_vals[0]]])
            explanation_bd = "Le matrici \\( B \\) e \\( D \\) sono calcolate per riprodurre la sequenza forzata."
            impulse_res_latex = ""

        self.C = sp.zeros(1, self.n)
        self.C[0, 0] = 1

        # Generazione HTML per le matrici con dettagli extra se c'è ingresso
        extra_info_html = ""
        if self.has_external_input:
            extra_info_html = f"""
            <div style="background: #e7f3ff; border: 1px solid #b3d7ff; border-radius: 8px; padding: 15px; margin-bottom: 20px; width: 95%; max-width: 700px; color: #004085;">
                <strong>Configurazione con Ingresso Esterno:</strong><br>
                Ingresso applicato: \( u(t) = {to_latex(self.input_expr_sym)} \)<br>
                Risposta all'impulso: {impulse_res_latex}<br>
                <span style="font-size: 0.9rem;">{explanation_bd}</span>
            </div>
            """


        
        html_content = f"""
        <div style="display: flex; flex-direction: column; gap: 25px; align-items: center; margin-top: 25px; width: 100%;">
            {extra_info_html}
            <p style="text-align: center; color: #666; font-style: italic;">Il sistema è realizzato in forma canonica di osservabilità.</p>

            
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; width: 95%; max-width: 700px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 6px solid #28a745;">
                <div style="font-weight: bold; color: #198754; margin-bottom: 15px; font-size: 1.2rem; display: flex; align-items: center;">
                    <span style="background: #28a745; color: white; border-radius: 4px; padding: 2px 8px; margin-right: 10px; font-size: 0.9rem;">A</span> 
                    Matrice di Stato (Dinamica)
                </div>
                <div style="font-size: 2rem; overflow-x: auto; padding: 15px 0; text-align: center; color: #212529;">
                    \\( A = {to_latex(self.A)} \\)
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; width: 95%; max-width: 700px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 6px solid #007bff;">
                <div style="font-weight: bold; color: #007bff; margin-bottom: 15px; font-size: 1.2rem; display: flex; align-items: center;">
                    <span style="background: #007bff; color: white; border-radius: 4px; padding: 2px 8px; margin-right: 10px; font-size: 0.9rem;">B</span> 
                    Matrice degli Ingressi (Controllo)
                </div>
                <div style="font-size: 2rem; overflow-x: auto; padding: 15px 0; text-align: center; color: #212529;">
                    \\( B = {to_latex(self.B)} \\)
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; width: 95%; max-width: 700px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 6px solid #fd7e14;">
                <div style="font-weight: bold; color: #fd7e14; margin-bottom: 15px; font-size: 1.2rem; display: flex; align-items: center;">
                    <span style="background: #fd7e14; color: white; border-radius: 4px; padding: 2px 8px; margin-right: 10px; font-size: 0.9rem;">C</span> 
                    Matrice delle Uscite (Osservazione)
                </div>
                <div style="font-size: 2rem; overflow-x: auto; padding: 15px 0; text-align: center; color: #212529;">
                    \\( C = {to_latex(self.C)} \\)
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; width: 95%; max-width: 700px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 6px solid #6c757d;">
                <div style="font-weight: bold; color: #495057; margin-bottom: 15px; font-size: 1.2rem; display: flex; align-items: center;">
                    <span style="background: #6c757d; color: white; border-radius: 4px; padding: 2px 8px; margin-right: 10px; font-size: 0.9rem;">D</span> 
                    Matrice di Legame Diretto
                </div>
                <div style="font-size: 2rem; overflow-x: auto; padding: 15px 0; text-align: center; color: #212529;">
                    \\( D = {to_latex(self.D_mat)} \\)
                </div>
            </div>
        </div>
        """
        self.matrices_html = html_content
        
        self.steps_latex.append({
            "title": "Costruzione Matrici dello Spazio di Stato",
            "content": html_content,
            "is_html": True
        })

    def solve(self, delta_specs_data, equation_str, input_expr_str=None):
        self.parse_input(delta_specs_data, equation_str, input_expr_str)
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
        input_expr = data.get("input_expr", "")
        result = solver.solve(delta_specs, equation_str, input_expr_str=input_expr)
        return jsonify({
            "success": True, 
            "data": result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
