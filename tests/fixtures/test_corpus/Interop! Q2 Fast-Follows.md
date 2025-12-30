# Interop: Q2 Fast-Follows

- [ ] Inventory the fast-follows and dependency relationships for Interop epics May 14, 2025

### Potential descope or backload

1. New Dealjackets from uniFI to Deal API; because "multi-partner" is undecided, Deal Central won't be consuming these and in fact we will not be even writing these deals into DealXG. We could optionally defer this functionality from [E42504.md](./E42504.md).

   1. From Voss: We will push new dealjackets, and are not yet concerned with the end-to-end or impact to Deal API consumers. 

2. New Dealjacket Events to Partners; in [E42504.md](./E42504.md), Finance Services is consuming more of our Fugu events to trigger data sync. The product requirements specify that these events are subsequently published via DTA Eventing. However, there are no immediate clients for these new events.