# Figure captions and slide text

## Figure 1

Tempo estimation accuracy on 50 held-out GTZAN-mini tracks (150 matched 8/15/30-second excerpts). Acc1 requires the raw estimate within 2% of the published reference; Acc2 additionally permits half/double tempo. Error bars are 95% track-bootstrap intervals. Ours has the highest observed Acc2; raw Acc1 is shown separately.

## Figure 2

Warm CPU file-to-BPM processing time on all six representative excerpts supported by the four displayed methods. Bars show medians and whiskers show P95 over 20 measurements per excerpt after five warm-ups, with one CPU thread and randomized interleaved method order. Ours and TempoCNN have comparable observed latency. All input loading, preprocessing and required output processing are included; startup and JIT warm-up are excluded.

## Review sentence

Our improved classical detector achieved the highest observed octave-tolerant tempo accuracy (Acc2: 86.7%) among the four evaluated methods, while matching the fastest neural baseline's observed CPU processing latency (13.17 ms vs. 13.61 ms for TempoCNN).

## Raw-accuracy qualification

Ours scored 62.7% Acc1 versus 64.0% for Beat This!; the highest-accuracy statement applies specifically to Acc2, not to every accuracy metric.
