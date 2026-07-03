"""
Example of source polarization and polar source coding in dit.

Source polarization (Arikan, 2010) applies the Arikan transform on the source
side: for ``N`` i.i.d. copies of a binary source ``X`` (optionally with side
information ``Y``), the synthesized coordinates ``U^N = X^N G_N`` split into
near-deterministic and near-uniform coordinates. Only the near-uniform
(high-entropy) coordinates need to be stored.
"""

import dit
from dit.coding import polar_source, source_polarization_profile
from dit.multivariate import gk_common_information, wyner_common_information


def main():
    # A doubly-symmetric binary source: X and its side information Y agree with
    # probability 0.9. The decoder has Y; the code stores only what Y cannot pin
    # down.
    dsbs = dit.Distribution(["00", "01", "10", "11"], [0.45, 0.05, 0.05, 0.45])

    print("Source-with-side-information H(X | Y) =", round(dit.shannon.conditional_entropy(dsbs, [0], [1]), 4))
    print()

    # The exact polarization profile for N = 4 copies: conditional entropies of
    # each synthesized coordinate given the past and all of Y. They sum to
    # N * H(X | Y) and spread toward 0 (determined) and 1 (uniform).
    profile = source_polarization_profile(
        dsbs,
        block_length=4,
        rv=0,
        crvs=[1],
        metrics=("entropy", "bhattacharyya"),
    )
    print("N = 4 source-polarization profile (given past and Y):")
    for row in profile:
        print(f"  U_{row['index']}:  H = {row['entropy']:.4f}   Z = {row['bhattacharyya']:.4f}")
    print(
        "  sum H =",
        round(sum(r["entropy"] for r in profile), 4),
        "= 4 * H(X | Y) =",
        round(4 * dit.shannon.conditional_entropy(dsbs, [0], [1]), 4),
    )
    print()

    # A lossless polar source code with the same side information at N = 8. The
    # rate is |high-entropy set| / N; decoding recovers each block exactly. The
    # finite-block set is chosen so that every dropped coordinate is deterministic
    # given the past and Y, which is what makes the round trip exact.
    code = polar_source(dsbs, block_length=8, rv=0, crvs=[1])
    print("N = 8 polar source code:")
    print("  high-entropy set:", code.high_entropy_set)
    print("  rate:", code.rate())

    # Encode/decode a matched (X, Y) block drawn from the source.
    block = [0, 1, 1, 0, 1, 0, 0, 1]
    side = [str(b) for b in block]
    encoded = code.encode(block)
    decoded = code.decode(encoded, side_information=side)
    print("  block   :", block)
    print("  encoded :", encoded)
    print("  decoded :", decoded, "(exact)" if decoded == block else "(mismatch)")
    print()

    # Goela et al. (2014) relate source polarization to common information. The
    # existing dit common-information measures quantify the shared randomness the
    # polar transform is reorganizing.
    print("Common information of the source (X ; Y):")
    print("  Gacs-Korner K =", round(gk_common_information(dsbs, [[0], [1]]), 4))
    print("  Wyner       C =", round(wyner_common_information(dsbs, [[0], [1]]), 4))


if __name__ == "__main__":
    main()
