from geodepoly import evaluate_hyper_catalan, evaluate_quadratic_slice, catalan_number


def main():
    # Quadratic slice: compare S[{t2}] with sum Catalan_n t2^n
    t2 = 0.05
    approx = evaluate_quadratic_slice(t2, max_weight=30)
    series = sum(catalan_number(n) * (t2 ** n) for n in range(0, 20))
    print("Quadratic slice S[t2]:", approx)
    print("Catalan series (20 terms):", series)

    # Small multivariate evaluation
    tv = {2: 0.05, 3: 0.01}
    print("S[t2=0.05, t3=0.01] (cutoff 10):", evaluate_hyper_catalan(tv, max_weight=10))


if __name__ == "__main__":
    main()


