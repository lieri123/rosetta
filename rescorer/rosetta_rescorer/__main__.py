"""Command line for the rescorer.

    python -m rosetta_rescorer serve   --db rosetta.db
    python -m rosetta_rescorer learn   --db rosetta.db --pairs corrections.jsonl
    python -m rosetta_rescorer ingest  --db rosetta.db notes/*.txt
    python -m rosetta_rescorer rescore --db rosetta.db --text "the rnatrix"
    python -m rosetta_rescorer eval    --db rosetta.db --synthesize notes.txt
    python -m rosetta_rescorer stats   --db rosetta.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional, Sequence

from .engine import Engine
from .evaluate import (
    Sample,
    correction_pairs,
    evaluate,
    format_adaptation,
    load_pairs,
    plot_curve,
    split_samples,
    sweep_substitution_margin,
    synthesize,
)
from .lexicon import load_word_list, tokenize
from .rescore import Thresholds, Token
from .server import serve
from .store import Store


def _engine(args: argparse.Namespace) -> Engine:
    store = Store(args.db) if args.db else None
    thresholds = Thresholds()
    if getattr(args, "low_confidence", None) is not None:
        thresholds.low_confidence = args.low_confidence
    if getattr(args, "substitute_nats", None) is not None:
        thresholds.substitute_nats = args.substitute_nats
    return Engine(store=store, thresholds=thresholds)


def cmd_serve(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = _engine(args)
    print(
        f"models: {len(engine.models.lexicon)} words, "
        f"{engine.models.language_model.total_words} training tokens, "
        f"{engine.store.correction_count() if engine.store else 0} corrections",
        file=sys.stderr,
    )
    serve(engine, host=args.host, port=args.port)
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    engine = _engine(args)
    pairs = []
    with open(args.pairs, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pairs.append((record["predicted"], record["corrected"]))

    result = engine.learn(pairs)
    print(
        f"learned from {result.pairs} corrections "
        f"({result.substitutions} character edits), "
        f"lexicon now {result.lexicon_size} words"
    )
    for written, read_as, count, rate in engine.models.confusion.top_confusions(10):
        print(f"  {written!r} read as {read_as!r}: {count} times ({rate:.1%})")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    engine = _engine(args)
    total = 0
    for path in args.files:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            total += engine.ingest_text(handle.read())
    if args.word_list:
        added = engine.models.lexicon.add_background(load_word_list(args.word_list))
        print(f"background word list: {added} words")
    print(f"ingested {total} tokens; lexicon now {len(engine.models.lexicon)} words")
    return 0


def cmd_rescore(args: argparse.Namespace) -> int:
    engine = _engine(args)
    text = args.text if args.text else sys.stdin.read()
    tokens = [Token(text=word, confidence=args.confidence) for word in tokenize(text)]
    scored = engine.rescore(tokens)

    if args.json:
        print(json.dumps([token.to_dict() for token in scored], indent=2))
        return 0

    print(" ".join(token.text for token in scored))
    print()
    for token in scored:
        if token.tier == "none":
            continue
        marker = "RED  " if token.tier == "red" else "AMBER"
        suggestion = f" -> {token.suggestion}" if token.suggestion else ""
        print(f"  {marker} {token.original!r}{suggestion}  ({token.reason})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    engine = _engine(args)

    if args.synthesize:
        with open(args.synthesize, "r", encoding="utf-8") as handle:
            text = handle.read()
        samples: List[Sample] = synthesize(text, error_rate=args.error_rate, seed=args.seed)
        synthetic = True
    else:
        samples = load_pairs(args.pairs)
        synthetic = False

    train, test = split_samples(samples, train_fraction=args.train_fraction, seed=args.seed)
    if not test:
        train, test = [], samples

    if args.prime_prior:
        # The language prior is allowed to see the training half's true text:
        # that is exactly what happens in use, where every line you correct
        # joins the corpus. The test half stays unseen.
        for sample in train:
            engine.ingest_text(sample.truth_text)

    before = evaluate(engine, test, synthetic=synthetic)

    pairs = correction_pairs(train)
    if pairs:
        engine.learn(pairs)
    after = evaluate(engine, test, synthetic=synthetic)

    if args.json:
        print(json.dumps(
            {"cold": before.to_dict(), "adapted": after.to_dict(), "corrections": len(pairs)},
            indent=2,
        ))
    else:
        print(format_adaptation(before, after, len(pairs)))

    if args.sweep:
        values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 15.0]
        curve = sweep_substitution_margin(engine, test, values)
        print("\nsubstitution margin sweep (adapted model)")
        for value, cer in curve:
            print(f"  margin {value:>5.1f} nats -> CER {cer:.4f}")
        if args.plot:
            if plot_curve(curve, args.plot, baseline=after.baseline_cer):
                print(f"\nwrote {args.plot}")
            else:
                print("\nmatplotlib not installed; skipped the plot", file=sys.stderr)

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    engine = _engine(args)
    print(json.dumps(engine.stats(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosetta_rescorer",
        description="Noisy-channel post-processing for handwriting recognition.",
    )
    parser.add_argument("--db", default="rosetta.db", help="SQLite database (shared with the service)")
    parser.add_argument("--low-confidence", type=float, default=None)
    parser.add_argument("--substitute-nats", type=float, default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", help="run the HTTP service")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8801)
    serve_parser.set_defaults(func=cmd_serve)

    learn_parser = sub.add_parser("learn", help="fold corrections into the models")
    learn_parser.add_argument("--pairs", required=True, help="JSONL of {predicted, corrected}")
    learn_parser.set_defaults(func=cmd_learn)

    ingest_parser = sub.add_parser("ingest", help="train the prior on known-good text")
    ingest_parser.add_argument("files", nargs="+")
    ingest_parser.add_argument("--word-list", help="optional background word list")
    ingest_parser.set_defaults(func=cmd_ingest)

    rescore_parser = sub.add_parser("rescore", help="rescore text from the command line")
    rescore_parser.add_argument("--text")
    rescore_parser.add_argument("--confidence", type=float, default=0.8)
    rescore_parser.add_argument("--json", action="store_true")
    rescore_parser.set_defaults(func=cmd_rescore)

    eval_parser = sub.add_parser("eval", help="measure against the raw recogniser")
    eval_parser.add_argument("--pairs", help="JSONL of {truth, observed}")
    eval_parser.add_argument("--synthesize", metavar="TEXT_FILE",
                             help="corrupt this text through a synthetic error model instead")
    eval_parser.add_argument("--train-fraction", type=float, default=0.5,
                             help="share of samples used to adapt the models (rest is held out)")
    eval_parser.add_argument("--prime-prior", action="store_true",
                             help="also train the language prior on the training half's true text")
    eval_parser.add_argument("--error-rate", type=float, default=0.08)
    eval_parser.add_argument("--seed", type=int, default=11)
    eval_parser.add_argument("--sweep", action="store_true")
    eval_parser.add_argument("--plot", metavar="PNG")
    eval_parser.add_argument("--json", action="store_true")
    eval_parser.set_defaults(func=cmd_eval)

    stats_parser = sub.add_parser("stats", help="show what the models have learned")
    stats_parser.set_defaults(func=cmd_stats)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "eval" and not args.pairs and not args.synthesize:
        parser.error("eval needs either --pairs or --synthesize")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
