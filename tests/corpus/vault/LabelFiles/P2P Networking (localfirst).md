---
tags:
  - document 📑
---
# P2P Networking (localfirst)

webRTC

[Overlay Network Implementations.md](./Overlay%20Network%20Implementations.md)

Magic Wormhole

Requires we trust some infra run by who knows who, or run it yourself <https://github.com/magic-wormhole>

Sendfiles.dev

Requires a bunch of infra and is mostly for files but is OSS <https://github.com/jchorl/sendfiles>

## Side channel

NAT transversal requires some shared side channel. 

Tailscale: DERP

WebRTC: shared signaling channel



XMPP or IRC (Bitcoin used IRC at the beginning)

## NAT Traversal

pwnat (cli) <https://github.com/samyk/pwnat>

STUN/TURN

ICE

Interactive Connectivity Establishment (ICE)

DERP

[DERP (Detoured Encrypted Routing Protocol)](https://tailscale.com/blog/how-tailscale-works#encrypted-tcp-relays-derp), which is a general purpose packet relaying protocol. It runs over HTTP, which is handy on networks with strict outbound rules, and relays encrypted payloads based on the destination’s public key.

<https://tailscale.com/blog/how-nat-traversal-works>


