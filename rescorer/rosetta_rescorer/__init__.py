"""Noisy-channel post-processing for handwriting recognition output.

The recognition API returns more than a string: it returns per-token
confidence, and sometimes alternative candidates. Almost nothing downstream
uses either. Treating the API output as a noisy observation of what was
actually written lets us recover the more likely true text:

    P(true | observed)  is proportional to  P(observed | true) * P(true)

The first term is an error model fitted from the user's own corrections; the
second is a language prior built from the user's own writing. This package
holds both, the decoder that combines them, and the evaluation harness that
says whether the combination actually beats the raw API.
"""

__all__ = [
    "align",
    "confusion",
    "lexicon",
    "lm",
    "rescore",
    "store",
]

__version__ = "0.1.0"
