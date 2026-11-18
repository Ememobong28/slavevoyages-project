# Which links between voyages are informative? (groundwork for GraphPFN)

A graph model needs links between voyages. Here we test candidate links on recorded values only. For each rule we look at all linked pairs where both voyages have the value, and compare with random pairs. Table: `link_rules.csv`; script: `analysis/graph_links.py`.

| Link rule | Pairs | Same nationality group | Typical tonnage difference | Voyages with a link | Missing tonnage that has a known neighbour |
| --- | --- | --- | --- | --- | --- |
| Random pair (baseline) | 200,000 | 26.6% | 55.0% | n/a | n/a |
| Same ship name | 138,916 | 84.2% | 33.3% | 80.9% | 35.3% |
| **Same ship name, within 10 years** | 63,187 | **92.0%** | **10.5%** | 70.4% | 19.8% |
| Same ship name, within 30 years | 101,849 | 88.2% | 23.4% | 77.0% | 28.2% |
| Same first captain | 47,585 | 97.3% | 30.0% | 57.0% | 12.7% |
| Same first captain, within 10 years | 34,470 | 98.8% | 22.8% | 53.0% | 8.7% |
| Same first owner | 81,638 | 99.0% | 33.3% | 46.6% | 9.4% |
| Same first owner, within 10 years | 66,578 | 99.0% | 30.2% | 44.3% | 8.1% |
| Same departure port | 18,490 | 89.1% | 48.2% | 26.0% | 11.9% |
| Same departure port, within 5 years | 5,568 | 92.8% | 40.4% | 9.5% | 3.0% |

(Groups with more than 40 voyages are sampled, not fully paired. Ship name, captain and owner are compared as upper-case text.)

## What it says
1. **For tonnage, the same ship within 10 years is the link to use.** Linked voyages differ by 10.5% typically, against 55% for random pairs, and better than the 17% our models reach. Widening to 30 years loses most of that (23.4%), so ships change over time (rebuilt, remeasured, or a different ship with the same name).
2. **For nationality, almost any ship, captain or owner link works.** 92-99% share a nationality group, against 27% at random. Owner and captain are cleaner for nationality (99%) but cover fewer voyages.
3. **Departure port is a weak link** for tonnage (40-48%) and covers little once restricted by time. Ports are already a column the models use.
4. **Coverage is the limit.** Even the best tonnage link (same ship, within 10 years) reaches a known neighbour for only 19.8% of voyages with no tonnage. Without the time limit it is 35.3%, with weaker agreement. Voyages with no linked neighbour get no help from the graph.

## Cautions
- Same name does not always mean same ship. Names recur across ships; the time limit reduces this but cannot remove it.
- Only the first captain and first owner are used; later ones are ignored for now.
- Linking through a value that is the target would leak the answer. The nationality graph must not link through nationality, and the final test must hide the voyage's own value. Linking through ship name, captain or owner is allowed.
- These are descriptive checks on recorded values; they do not yet show a model improves.

## Suggested first graph (to be decided with the professor)
Edges: same ship name within 10 years, plus same first owner, plus same first captain. Compare against a version with ship name only, and a version with no graph.
