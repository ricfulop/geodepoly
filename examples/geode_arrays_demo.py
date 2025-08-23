from geodepoly.hyper_catalan import bi_tri_array, hyper_catalan_coefficient


def main():
    A = bi_tri_array(4, 3)
    print("Bi–Tri array (partial):")
    for row in A:
        print(row)
    # Verify a spot match
    print("check A[2][1] == coef({2:2,3:1}):", A[2][1], hyper_catalan_coefficient({2: 2, 3: 1}))


if __name__ == "__main__":
    main()


