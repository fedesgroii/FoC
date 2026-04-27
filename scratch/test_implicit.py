import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

transformations = standard_transformations + (implicit_multiplication_application,)

try:
    expr = parse_expr("0.4x2", transformations=transformations)
    print(f"Parsed: {expr}")
except Exception as e:
    print(f"Error: {e}")
