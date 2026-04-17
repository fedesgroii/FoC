import sympy as sp
import re
from sympy.parsing.sympy_parser import parse_expr

def test_logic(y_input, time_type):
    print(f"\n--- Testing: {y_input} ({time_type}) ---")
    y_input_clean = y_input.replace('^', '**')
    constants_str = list(set(re.findall(r'C\d+', y_input_clean)))
    constants_str.sort(key=lambda x: int(x[1:]))
    n = len(constants_str)

    if time_type == 'continuous':
        t = sp.symbols('t', real=True)
    else:
        t = sp.symbols('t', integer=True)
        
    local_dict = {'t': t, 'e': sp.exp(1)}
    for c in constants_str:
        local_dict[c] = sp.symbols(c)

    y_expr = parse_expr(y_input_clean, local_dict=local_dict)
    
    equations = [y_expr]
    for i in range(1, n + 1):
        if time_type == 'continuous':
            equations.append(sp.diff(equations[-1], t))
        else:
            equations.append(y_expr.subs(t, t + i))
            
    y_syms = [sp.symbols(f"y_{i}") for i in range(n + 1)]
    sys_eqs = [sp.Eq(equations[i], y_syms[i]) for i in range(n)]
    C_syms = [local_dict[c] for c in constants_str]
    
    sol = sp.solve(sys_eqs, C_syms, dict=True)
    if not sol:
        print("Failed to solve!")
        return
    sol_dict = sol[0]
    
    y_n_expr = equations[n].subs(sol_dict)
    y_n_expr = sp.simplify(sp.expand(y_n_expr))
    
    coeffs = []
    for i in range(n):
        c = sp.simplify(y_n_expr.diff(y_syms[i]))
        coeffs.append(c)
        print(f"A[{n-1}, {i}] = {c}")
        
    u_term = sp.simplify(y_n_expr - sum(coeffs[i]*y_syms[i] for i in range(n)))
    print(f"u(t) = {u_term}")

test_logic("e^t*C1 + e^(2*t)*C2 + t*e^t", "continuous")
test_logic("e^t*cos(3*t)*C1 + e^t*sin(3*t)*C2", "continuous")
test_logic("e^t*(C1 + C2*t + C3*t**2)", "continuous")
test_logic("2^t*C1 + C2 + t", "discrete")
test_logic("(-1)^t*(C1 + t*C2) + t", "discrete")
