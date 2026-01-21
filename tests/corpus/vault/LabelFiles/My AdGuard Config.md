# My AdGuard Config

## Braddock

AdGuard Pro

> for some reason, this is not enrolled into our account and I probably pay extra for this license

All main options enabled

Local DNS VPN - not native mode

Adguard DNS chosen as the resolver

Advanced settings enabled, so set the app in Fill Tunnel mode.

Fallback DNS server is set to “none”, otherwise the system default will engage in failure cases. Note this can also happen in Split Tunnel mode if the vpn is flaky (iOS will try and protect the user by rerouting)

Ref: [Local IOS iPhone VPNs and iCloud Private Relay  .md](./Local%20IOS%20iPhone%20VPNs%20and%20iCloud%20Private%20Relay%20%20.md)

Note: if it stops working, the local vpn is crashing because there are too many rules - bypassing an iOS limit and killing the process. This may appear to work if I have Split Tunnel enabled, as iOS will fail over to some default unknown to me behavior.

Using some custom [DNS blocklists.md](./DNS%20blocklists.md) for TikTok and Amazon.