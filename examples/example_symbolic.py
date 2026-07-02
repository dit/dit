"""
Example of using dit to compute symbolic information measures.

Requires sympy (``pip install dit[symbolic]``).
"""

import sympy

from dit.multivariate import coinformation, entropy, total_correlation
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


if __name__ == "__main__":
    main()
