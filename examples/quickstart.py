from geodepoly import solve_poly

# Solve x^3 - 6x^2 + 11x - 6 = 0
coeffs = [-6, 11, -6, 1]
roots = solve_poly(coeffs, method="hybrid", resum="pade")
print("roots:", roots)
