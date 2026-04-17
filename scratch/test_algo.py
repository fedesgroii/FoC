import re
import sympy as sp

def test(y_input, time_type):
    t = sp.symbols('t')
    
    # PASSO 1: Analisi costanti
    constants = list(set(re.findall(r'C\d+', y_input)))
    constants.sort(key=lambda x: int(x[1:]))
    n = len(constants)
    
    expr_str = y_input.replace('e^', 'exp').replace('^', '**')
    # replace e with E if it's standalone, but exp() is better handled by just replacing e^
    expr_str = re.sub(r'\be\b', 'E', expr_str)
    
    y_expr = sp.sympify(expr_str)
    const_syms = [sp.symbols(c) for c in constants]
    
    # PASSO 2: Generazione equazioni
    equations = [y_expr]
    if time_type == 'continuous':
        for i in range(1, n+1):
            equations.append(sp.diff(equations[-1], t))
        y_vars = [sp.Function('y')(t)] + [sp.Derivative(sp.Function('y')(t), t, i) for i in range(1, n+1)]
    else:
        for k in range(1, n+1):
            equations.append(y_expr.subs(t, t+k))
        y_vars = [sp.Function('y')(t+k) for k in range(n+1)]
        
    print(f"Equations: {equations}")
        
    # PASSO 3: Isolamento costanti
    system_eqs = []
    # y_vars[i] will be used as the variable representing equations[i]
    # actually, solving for C1...Cn in terms of y, y',...
    # we can use abstract symbols for y_vars during solve to avoid issues
    y_syms = sp.symbols(f'Y0:{n}')
    for i in range(n):
        expr = equations[i].expand()
        # Separate terms with constants and without
        terms = expr.as_ordered_terms()
        unknown_terms = [term for term in terms if any(c.name in str(term) for c in const_syms)]
        known_part = expr - sum(unknown_terms)
        system_eqs.append(sp.Eq(sum(unknown_terms), y_syms[i] - known_part))
        
    print(f"System Eqs: {system_eqs}")
    constants_solution = sp.solve(system_eqs, const_syms)
    print(f"Constants Sol: {constants_solution}")
    
    # PASSO 4: Costruzione equazione finale
    n_th_equation = equations[n]
    for const, expr in constants_solution.items():
        n_th_equation = n_th_equation.subs(const, expr)
        
    final_equation = sp.simplify(y_syms[-1] if False else n_th_equation) # wait, it's Y_n = n_th_equation(C1..Cn replaced)
    # let's write it properly:
    Y_n = sp.symbols(f'Y{n}')
    final_expr = n_th_equation.subs(constants_solution)
    final_eq = sp.Eq(Y_n, sp.simplify(final_expr))
    print(f"Final Eq: {final_eq}")
    
    # Riscrivi nella forma Y_n + a_{n-1} Y_{n-1} ... + a_0 Y_0 = b u(t)
    # Rearrange final_expr: Y_n - final_expr = 0
    diff_eq = sp.expand(Y_n - final_expr)
    print(f"Diff Eq: {diff_eq}")
    
    # Extract coefficients
    coeffs = []
    for i in range(n):
        coeffs.append(diff_eq.coeff(y_syms[i]))
        
    # the input u(t) term is the rest
    input_term = diff_eq.subs({y_syms[i]: 0 for i in range(n)}).subs(Y_n, 0)
    # actually input_term is the known terms
    print(f"Coeffs: {coeffs}")
    print(f"Input term: {-input_term}")

test("exp(t)*C1 + exp(2*t)*C2", "continuous")
print("---")
test("2**t*C1 + C2 + t", "discrete")

