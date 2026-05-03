from flask import Blueprint, request, jsonify
import sympy as sp
import re
from sympy import Matrix, symbols, simplify, sympify, Eq, solve, nsolve
from ast import literal_eval
from sympy.parsing.sympy_parser import parse_expr
from .utils import transformations, sostituisci_pedici, formatta_x_e

linearizzazione_bp = Blueprint("linearizzazione", __name__)

@linearizzazione_bp.route('/api/linearizzazione', methods=['POST'])
def linearizzazione():
    try:
        # Estrai i dati dal JSON
        data = request.get_json()
        equazioni = data.get('equazioni', [])
        equazione_uscita = data.get('equazioneUscita', '')
        valore_ingresso_str = data.get('valoreIngresso', '0')
        tipo_dominio = data.get('dominio', '')


        valore_ingresso = sympify(valore_ingresso_str)
        print("DEBUG: valore_ingresso =", valore_ingresso)

        numero_equazioni = len(equazioni)

        # Definisci le variabili simboliche x1, x2, ..., xn
        x = symbols(f'x1:{numero_equazioni+1}')  # Genera x1, x2, ..., xn
        u = symbols('u')

        # Pre-elabora le equazioni
        eqs_sostituite = []
        for eq in equazioni:
            try:
                # Estrai il RHS se è presente un '='
                if '=' in eq:
                    eq = eq.split('=')[-1].strip()
                
                # Rimuove (t+1), (t), ecc.
                eq = re.sub(r'\(t\+1\)', '', eq)
                eq = re.sub(r'\(t\)', '', eq)
                
                eq_modificata = sostituisci_pedici(eq)
                local_dict = {f'x{i+1}': x[i] for i in range(numero_equazioni)}
                if numero_equazioni == 1:
                    local_dict['x'] = x[0]
                local_dict['u'] = u
                print("DEBUG: local_dict =", local_dict)
                expr = parse_expr(eq_modificata, transformations=transformations, local_dict=local_dict)

                # Sostituisci u con il valore di ingresso
                expr = expr.subs(u, valore_ingresso)

                # mapping/subs non più necessario grazie a local_dict
                # mapping = {sp.Symbol(f'x{i+1}'): x[i] for i in range(numero_equazioni)}
                # expr = expr.subs(mapping)

                eqs_sostituite.append(expr)
            except Exception as e:
                return jsonify({
                    "success": False,
                    "errore": f"Errore nell'elaborazione dell'equazione '{eq}': {str(e)}"
                })

        print("DEBUG: eqs_sostituite =", eqs_sostituite)

        # Costruisci le equazioni per il punto di equilibrio
        if tipo_dominio == 'typeContinuo':
            eqs_punto_di_equilibrio = [Eq(eq, 0) for eq in eqs_sostituite]
        elif tipo_dominio == 'typeDiscreto':
            eqs_punto_di_equilibrio = [Eq(x[i], eq) for i, eq in enumerate(eqs_sostituite)]
        else:
            return jsonify({"success": False, "errore": "Selezionare un tipo di dominio valido."})

     

        # DEBUG: stampa il sistema di equazioni in forma Python e LaTeX
        print("DEBUG: Risolvo il sistema:")
        for eq in eqs_punto_di_equilibrio:
            print("   ", eq, "  (LaTeX:", sp.latex(eq), ")")
        # Risolvi il sistema rispetto alle variabili di stato x
        try:
            soluzioni_raw = solve(eqs_punto_di_equilibrio, list(x), dict=True)
        except Exception as e:
            print(f"DEBUG: solve() failed: {e}")
            soluzioni_raw = []

        # Fallback per equazioni trascendenti (es. x = sin(x)) se solve non trova nulla o fallisce
        if not soluzioni_raw:
            print("DEBUG: Provo con nsolve (fallback numerico)")
            vars_list = list(x)
            exprs = [eq.lhs - eq.rhs for eq in eqs_punto_di_equilibrio]
            
            # Punti di partenza per nsolve
            for x0_val in [0, 1, -1, 5, -5]:
                try:
                    guess = [x0_val] * len(vars_list)
                    root = nsolve(exprs, vars_list, guess)
                    # Converti root (Matrix) in dizionario
                    sol_dict = {vars_list[i]: root[i] for i in range(len(vars_list))}
                    
                    # Evita duplicati numerici
                    is_duplicate = False
                    for s_exist in soluzioni_raw:
                        if all(abs(float(sol_dict[v] - s_exist[v])) < 1e-6 for v in vars_list):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        soluzioni_raw.append(sol_dict)
                except Exception:
                    continue

        # Filtra solo soluzioni reali (ignora quelle con parte immaginaria)
        soluzioni = []
        for sol in soluzioni_raw:
            valido = True
            for val in sol.values():
                try:
                    val_f = val.evalf()
                    if hasattr(val_f, 'is_real') and val_f.is_real is False:
                        valido = False
                except:
                    # Se non può essere valutato numericamente, assumiamo sia simbolico valido per ora
                    pass
            if valido:
                soluzioni.append(sol)

        # Blocco diagnostico per stampare il contenuto delle equazioni e delle soluzioni
        print("Equazioni punto di equilibrio:", eqs_punto_di_equilibrio)
        print("Soluzioni trovate:", soluzioni)
        # Migliora la capacità di trovare soluzioni simboliche parametriche o, in caso estremo, ricade su fallback manuale
        if not soluzioni:
            # Tenta una soluzione simbolica con solve(..., set=True) per ottenere relazioni parametriche
            sol_set = solve(eqs_punto_di_equilibrio, dict=True, set=True)
            if isinstance(sol_set, tuple) and len(sol_set) == 2 and sol_set[1]:
                soluzione_param = list(sol_set[1])[0]
                soluzioni = [soluzione_param]
            else:
                # fallback: crea simboli liberi
                parametri_liberi = [s for s in x if all(s not in eq.free_symbols for eq in eqs_punto_di_equilibrio)]
                simboli_parametrici = symbols(f'c1:{len(parametri_liberi)+1}')
                sol_dinamica = {s: simboli_parametrici[i] for i, s in enumerate(parametri_liberi)}
                for i, s in enumerate(x):
                    if s not in sol_dinamica:
                        sol_found = solve(eqs_punto_di_equilibrio, s, dict=False)
                        sol_dinamica[s] = sol_found[0] if sol_found else 0
                soluzioni = [sol_dinamica]

        # Nuovo blocco: accetta anche soluzioni simboliche, mantenendo consistenza
        soluzioni_dict_raw = []
        for sol in soluzioni:
            sol_dict = {}
            for i, var in enumerate(x):
                chiave = f"x_{i+1}"
                valore = sol.get(var, None)
                if valore is None:
                    # parametro libero: creiamo un Symbol c_{i+1}
                    sol_dict[chiave] = sp.Symbol(f"c_{i+1}")
                else: 
                    # semplifichiamo e manteniamo forme simboliche corrette (es. \pi invece di approssimare a rational)
                    sol_dict[chiave] = sp.simplify(valore)
            soluzioni_dict_raw.append(sol_dict)

        # Sostituisci simboli con espressioni semplici se sono riferimenti ad altri simboli o costanti (correggendo logica "v in sol")
        for sol_dict in soluzioni_dict_raw:
            subs_map = {sp.Symbol(k): v for k, v in sol_dict.items()}
            changed = True
            while changed:
                changed = False
                for k, v in sol_dict.items():
                    if hasattr(v, 'subs'):
                        new_v = sp.simplify(v.subs(subs_map))
                        if new_v != v:
                            sol_dict[k] = new_v
                            subs_map[sp.Symbol(k)] = new_v
                            changed = True

        # Validazione delle soluzioni (scarta quelle che non azzerano il sistema originale entro 1e-10)
        soluzioni_reali = []
        for sol_dict in soluzioni_dict_raw:
            # Crea un dizionario di sostituzione per le variabili rispetto alle equazioni originali
            subs_eq_dict = {x[i]: sol_dict[f"x_{i+1}"] for i in range(numero_equazioni)}
            
            is_valid = True
            # Controlliamo solo se la soluzione numerica è calcolabile (no parametri c_ liberi)
            # hasattr(v, "free_symbols") is implicit for SymPy expressions
            if not any(c.name.startswith("c_") for v in sol_dict.values() if hasattr(v, "free_symbols") for c in v.free_symbols if hasattr(c, "name")):
                for eq in eqs_punto_di_equilibrio:
                    expr = eq.lhs - eq.rhs
                    residuo = expr.subs(subs_eq_dict)
                    try:
                        # Valuta il residuo numericamente con tolleranza 1e-10
                        residuo_f = float(sp.N(residuo, 15))
                        if abs(residuo_f) > 1e-10:
                            is_valid = False
                            break
                    except TypeError:
                        # Se TypeError, l'espressione contiene ancora incognite, non scartiamo
                        pass
            
            if is_valid:
                soluzioni_reali.append(sol_dict)

        # Step 1: Punto di equilibrio
        # Build LaTeX for equilibrium equations and solution
        eqs_latex = " \\\\ ".join([sp.latex(eq) for eq in eqs_punto_di_equilibrio])
        latex_steps = []
        for idx, sol in enumerate(soluzioni_reali):
            if tipo_dominio == "typeContinuo":
                descrizione_dominio = r"\mathbb{T} = \mathbb{R}\implies f(\mathbf{x}_e, u_e) = 0 \implies "
            else:
                descrizione_dominio = r"\mathbb{T} = \mathbb{Z}\implies f(\mathbf{x}_e, u_e) = \mathbf{x}_e \implies "

            # costruisci dizionario di sostituzioni per x_i → c_i
            subs_dict = { x[j]: sp.Symbol(f"c_{j+1}") for j in range(numero_equazioni) }
            descrizione_latex = (
                descrizione_dominio
                + r"\mathbf{x}_e = \left("
                + ", ".join(formatta_x_e(v.subs(subs_dict)) for v in sol.values())
                + r"\right)"
            )

            # Rileva tutti i simboli c_i anche dentro espressioni complesse
            const_set = {c for v in sol.values() for c in v.free_symbols if c.name.startswith("c_")}
            if const_set:
                const_latex = ", ".join(sp.latex(c) for c in sorted(const_set, key=lambda s: s.name))
                descrizione_latex += " \\ \\operatorname{con}\\," + const_latex + " \\in \\mathbb{R}"

            latex_steps.append({
                "title": f"Punto di equilibrio {idx + 1}:",
                "content": descrizione_latex,
                "overflow": True
            })

        # Verifica se ci sono soluzioni reali
        if not soluzioni_reali:
            return jsonify({
                "success": False,
                "errore": "Nessuna soluzione reale trovata per il punto di equilibrio.",
                "suggerimento": "Verifica che: 1) Le equazioni siano corrette 2) Il valore di ingresso sia compatibile 3) Il sistema ammetta soluzioni reali",
                "debug_soluzioni_raw": [str(s) for s in soluzioni]
            })

        soluzione_latex = ",\\quad ".join([f"{k} = {formatta_x_e(v)}" for k, v in soluzioni_reali[0].items()])


        from sympy import diff, Matrix
        def matrix_to_latex(mat):
            mat = sp.nsimplify(mat, rational=True)
            rows = [" & ".join(sp.latex(el) for el in row) for row in mat.tolist()]
            return "\\begin{bmatrix}" + " \\\\ ".join(rows) + "\\end{bmatrix}"

        for idx, sol in enumerate(soluzioni_reali):
            punto_eq_subs = {}
            for i in range(numero_equazioni):
                chiave = f"x_{i+1}"
                valore = sol[chiave]
                punto_eq_subs[x[i]] = valore
            punto_eq_subs[u] = valore_ingresso

            try:
                f_exprs = []
                for eq in equazioni:
                    # Stessa pulizia fatta sopra
                    if '=' in eq:
                        eq = eq.split('=')[-1].strip()
                    eq = re.sub(r'\(t\+1\)', '', eq)
                    eq = re.sub(r'\(t\)', '', eq)

                    eq_modificata = sostituisci_pedici(eq)
                    local_dict = {f'x{i+1}': x[i] for i in range(numero_equazioni)}
                    if numero_equazioni == 1:
                        local_dict['x'] = x[0]
                    local_dict['u'] = u
                    expr = parse_expr(eq_modificata, transformations=transformations, local_dict=local_dict)
                    f_exprs.append(expr)

                A = Matrix([[diff(expr, x_var) for x_var in x] for expr in f_exprs]).subs(punto_eq_subs).applyfunc(sp.simplify)
                B = Matrix([[diff(expr, u)] for expr in f_exprs]).subs(punto_eq_subs).applyfunc(sp.simplify)

                # Pulizia equazione uscita
                h_uscita = equazione_uscita
                if '=' in h_uscita:
                    h_uscita = h_uscita.split('=')[-1].strip()
                h_uscita = re.sub(r'\(t\)', '', h_uscita)

                h_uscita_mod = sostituisci_pedici(h_uscita)
                local_dict = {f'x{i+1}': x[i] for i in range(numero_equazioni)}
                if numero_equazioni == 1:
                    local_dict['x'] = x[0]
                local_dict['u'] = u
                h_orig = parse_expr(h_uscita_mod, transformations=transformations, local_dict=local_dict)
                
                C = Matrix([[diff(h_orig, x_var) for x_var in x]]).subs(punto_eq_subs).applyfunc(sp.simplify)
                D = Matrix([[diff(h_orig, u)]]).subs(punto_eq_subs).applyfunc(sp.simplify)

                contenuto_matrici = (
                    r"\begin{gathered} "
                    r"A = " + matrix_to_latex(A) + r",\quad B = " + matrix_to_latex(B) + r" \\ " +
                    r"C = " + matrix_to_latex(C) + r",\quad D = " + matrix_to_latex(D) +
                    r" \end{gathered}"
                )

                # Usa lo stesso subs_dict di sopra per sostituire x_i → c_i
                xe_latex = ", ".join(formatta_x_e(v.subs(subs_dict)) for v in sol.values())
                title = f"Linearizzazione relativa a \\( \\mathbf{{x}}_e = \\left({xe_latex}\\right) \\):"
                latex_steps.append({
                    "title": title,
                    "content": contenuto_matrici
                })

            except Exception as e:
                return jsonify({
                    "success": False,
                    "errore": f"Errore durante la linearizzazione del punto {idx+1}: {str(e)}"
                })

        return jsonify({
            "success": True,
            "latex": latex_steps,
            "messaggio": "Punto di equilibrio calcolato correttamente.",
            "punto_equilibrio": [{k: str(v) for k, v in sol.items()} for sol in soluzioni_reali],
            "punto_equilibrio_tex": soluzione_latex,
            "valore_ingresso": str(valore_ingresso),
            "equazioni": equazioni,
            "tipo_dominio": tipo_dominio,
            "numero_equazioni": numero_equazioni
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "errore": f"Errore durante il calcolo: {str(e)}"
        })