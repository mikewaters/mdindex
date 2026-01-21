# Setting up Private Local DNS 

Depends On:: 

- Bypass the Optimum modem for IP Routing

   - also requires that I reprovision wifi, which will be a PITA

- Provision a pi-hole

   - I can use the NAS/Docker for now, but I will need some dedicated device/rpi

## Options

### ✅ OPNSense

<https://docs.opnsense.org>

OPNSense is a FreeBSD based system that uses Unbound DNS resolver service. This is officially integrated as a provider in Tailscale, and so is probably the solution I will choose. I will need to host it somewhere locally.

[Tailscale and OPNSense](https://tailscale.com/kb/1299/opnsense-unbound)

#### OPNSense Hardware Recommendations

- Protectli firewall/routers <https://protectli.com/product/fw2b/>

- Zimaboard (more of a DIY option) <https://www.zimaspace.com>

Discussions:

- [Hardware and Performance](https://forum.opnsense.org/index.php?board=21.0) forum for opnsense

- [Lots of options, from 2024](https://homenetworkguy.com/review/opnsense-hardware-recommendations/)

- [Discussion in opnsense forum](https://forum.opnsense.org/index.php?topic=38030.0) from 2024

### Use pi-hole DNS + mDNS service

Requires that I can control the DNS Server settings for all devices directly. If I can set this at the DHCP server/router it would be better.

### Run something locally for dev servers

Use a tool like \[localalias\](<https://github.com/peterldowns/localias>) to map dns names to localhost or LAN services. Might be good for development, as it handles TLS. This doesnt help me on phones or iot devices though.

Using a router to force DNS requests, even when devices want to bypass it

<https://den.dev/blog/pihole/>

### Use Tailscale 