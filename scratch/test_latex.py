import sympy as sp

n = 2
Y_n = sp.symbols('Y2')
Y0, Y1 = sp.symbols('Y0 Y1')
final_expr = 2*Y0 - 3*Y1
latex_str = sp.latex(final_expr).replace('Y', 'y')
print(latex_str)
y_latex = ['y(t)', '\\dot{y}(t)', '\\ddot{y}(t)']
for i in range(n):
    latex_str = latex_str.replace(f'y_{{{i}}}', y_latex[i])
print(latex_str)
