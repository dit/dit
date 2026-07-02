"""
Example of using dit to compute symbolic information measures.

Requires sympy (``pip install dit[symbolic]``).
"""

import sympy

from dit.multivariate import coinformation, entropy, total_correlation, wyner_common_information
from dit.pid import PID_WB
from dit.symbolic import simplify, symbolic_distribution, symbols


def main():
    p = symbols("p")

    # A symbolic coin: H(p) is the binary entropy function.
    coin = symbolic_distribution(["0", "1"], [p, 1 - p])
    print("H(coin) =", entropy(coin))
    print("H(coin) at p=1/2 =", entropy(coin).subs(p, sympy.Rational(1, 2)))
    print()

    # A "giant bit": two perfectly correlated bits. H(XY) = I(X:Y) = T = H(p).
    giant_bit = symbolic_distribution(["00", "11"], [p, 1 - p])
    print("H(XY) =", simplify(entropy(giant_bit)))
    print("T(X:Y) =", simplify(total_correlation(giant_bit)))
    print("I(X:Y) =", simplify(coinformation(giant_bit)))
    print()

    # PID of a 3-variable giant bit: purely redundant.
    gb3 = symbolic_distribution(["000", "111"], [p, 1 - p])
    pid = PID_WB(gb3)
    print("I_min redundancy {0}{1} =", simplify(pid.get_pi(((0,), (1,)))))
    print("I_min unique {0} =", pid.get_pi(((0,),)))
    print("I_min synergy {01} =", pid.get_pi(((0, 1),)))
    print()

    # Optimization-based common information, symbolically. The Wyner common
    # information of a doubly-symmetric binary source closes in radicals:
    # 1 + h(a) - 2 h(a0) with a0 (1 - a0) = a / 2.
    a = symbols("a")
    dsbs = symbolic_distribution(["00", "01", "10", "11"], [(1 - a) / 2, a / 2, a / 2, (1 - a) / 2])
    print("Wyner CI(DSBS) =", wyner_common_information(dsbs, backend="symbolic"))


if __name__ == "__main__":
    main()
