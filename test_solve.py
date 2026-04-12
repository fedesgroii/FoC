import sympy as sp
x1 = sp.Symbol('x1')
u = sp.Symbol('u')
eq1 = (x1-1)*(x1-2) + (x1-3)*u
# in the script it replaces u=0 early for eqs_punto_di_equilibrio
eq1 = eq1.subs(u, 0)
print("Generico:", sp.solve([sp.Eq(eq1, 0)], dict=True))
print("Esplicito:", sp.solve([sp.Eq(eq1, 0)], [x1], dict=True))
