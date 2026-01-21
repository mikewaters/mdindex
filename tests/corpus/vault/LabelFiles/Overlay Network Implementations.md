---
tags:
  - landscape
Document Type:
  - Technical
Topic:
  - software
Subject:
  - Solution
---
# Overlay Network Implementations

## Netbird

<https://github.com/netbirdio/netbird>

> NetBird creates a WireGuard-based overlay network that automatically connects your machines over an encrypted tunnel, leaving behind the hassle of opening ports, complex firewall rules, VPN gateways, and so forth

> > **NetBird combines a configuration-free peer-to-peer private network and a centralized access control system in a single platform, making it easy to create secure private networks for your organization or home.**

## Tailscale and ecosystem

# Tailscale and Alternatives for and Overlay Network

Why?

Encryption on the wire, regardless of protocol (database, http, etc) where no provider has decryption keys (eg no TLS termination by cloudflare). 

Expose a mesh overlay to connect hosts in multiple networks including private/home/NATed and cellular.

Removes the need for every host to manage its own encryption point to point and instead encrypt at the network interface.

Prefer Wireguard to VPN.

Do not rely on this service for DNS

# Implementations

### Tailscale

<https://tailscale.com/blog/how-tailscale-works>

### Headscale

### OpenZiti

<https://openziti.io/>

### Zero tier

### Nebula

Nebula <https://github.com/slackhq/nebula>

### Netbird

<https://yggdrasil-network.github.io/>

### Twingate

> Twingate uses a service-based access model, rather than host/IP/ACL-based, as Wireguard defines the world.

> 

### Firezone

> Enterprise-ready zero-trust access platform built on WireGuard

source: [GitHub - firezone:firezone: Enterprise-ready zero-trust access platform built on WireGuard®..html](https://github.com/firezone/firezone)

<https://news.ycombinator.com/item?id=41173330>

Compared to Tailscale:

> The main difference is how access is managed. Instead of configuring ACLs, you define policies which are a 1:1 mapping between a user group (manually created or synced from your IdP) and the resource you want to allow access for. Another difference is how our load balancing / failover system works - it's automatic across all the Gateways in a particular Site.

## Notes

search HN



> Tailscale is open source, which means that the source code is available for anyone to review and audit for security vulnerabilities. ZeroTier, on the other hand, is proprietary, which means that the source code is not publicly available.

source: [Tailscale vs ZeroTier: A Comprehensive Comparison of Two Popular VPN Solutions - E2Encrypted.html](https://www.e2encrypted.com/posts/tailscale-vs-zerotier-comprehensive-comparison/)



Netbird (golang)

<https://github.com/netbirdio/netbird>