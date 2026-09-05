# Lichess Dataset Plan

## Corpus analysis

This plan is based on the complete official Lichess puzzle export dated 2026-07-05, containing 6,057,356 puzzles. All rows have usable FEN, move, rating, and game URL fields.

- Mean puzzle rating: 1,472.4; standard deviation: 547.4.
- 1,351,407 puzzles (22.3%) are below rating 1,000, so uniform sampling would overemphasize beginner material.
- 149,797 puzzles (2.5%) are rated 2,600 or above; only 943 meet the quality filter below, so this tier is deliberately present but small.
- Mean solution-line length is 4.63 plies. The largest group is 4-5 plies (3,072,603 puzzles), followed by 6-7 plies (1,588,784).
- Common overlapping themes are `short`, `endgame`, `middlegame`, `crushing`, `mate`, `advantage`, and `fork`. The builder uses coarse theme families to avoid mate/endgame dominance.

## Quality filter

Keep puzzles with all of:

- `RatingDeviation <= 80`
- `Popularity >= 50`
- `NbPlays >= 100`

This leaves 3,147,053 community-vetted puzzles. Rating is not used as a quality filter; it is used only for balanced sampling.

## Split design

Use the first tactical solution position only: apply the opponent's blunder (the first UCI move), then ask the model for a move from that FEN. This gives one source position per puzzle and avoids overweighting long puzzle lines.

Whole Lichess source games are deterministically assigned to one split using a hash of their game ID. The allocation is 68% SFT, 20% GRPO, and 12% validation. Canonical FENs are additionally deduplicated across the ordered splits, so validation contains no board position seen by SFT or GRPO.

## Generated datasets

- `sft_train.jsonl`: 60,000 positions expanded to one row per legal UCI move. This is the legality-learning dataset.
- `grpo_train.jsonl`: 15,000 distinct, unseen-from-SFT positions with solution and metadata for Stockfish-rewarded GRPO.
- `validation.jsonl`: 10,000 held-out positions for legal-UCI rate, legal-move rate, solution rate, and engine-quality evaluation.

Difficulty is sampled with the following target shares: 6% under 1000, 9% 1000-1199, 13% 1200-1399, 16% 1400-1599, 15% 1600-1799, 15% 1800-1999, 12.1% 2000-2199, 8% 2200-2399, 5% 2400-2599, and 0.9% 2600+.

Within each rating tier, target theme-family shares are 30% mate, 35% tactical, 25% endgame, and 10% other. Sparse intersections are backfilled from the same rating tier, preserving a full dataset rather than silently dropping samples.
