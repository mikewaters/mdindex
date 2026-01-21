# Local IOS iPhone VPNs and iCloud Private Relay  



Verdict: Need to disable the local AdGuard pseudo-VPN and rely on system DNS outbound tunnel. Can still set this to use the AdGuard DNS resolver services, but blocking functionality is lost given AdGuard can no longer intercept direct IP traffic or any DNS lookups that might have been done by the system.

Note: Even if I run the local AdGuard pseudo VPN, iOS may choose to override this for local traffic - it has been demonstrated to do this. This will happen if Split Tunnel is used, given this is a feature provided by IOS - and if the VPN is flaky it will algorithmically fail over to the OS default (the other side of the split). And so using Full Tunnel prevents this, but also relies on the local von and the Adguard resolver working properly.

# How iOS & macOS route DNS when AdGuard Pro is installed

### 1\. Two very different DNS modes inside AdGuard Pro

1. Local-VPN (“AdGuard” implementation)  

   - The app creates a **personal VPN tunnel on‐device**. All traffic is sent through the tunnel so AdGuard can intercept every DNS packet and, if you enable it, filter other traffic too.  

2. Native DNS profile (“Native” implementation)  

   - The app simply installs an Apple configuration profile that tells the *system* which encrypted resolver to use. **No VPN tunnel is created.**

### 2\. Does Apple ever bypass AdGuard’s protection?

| Scenario | How iOS / macOS behaves | Effect on AdGuard protection | 
|---|---|---|
| AdGuard running in **Native** mode | DNS is handed to the system resolver, so **any feature that overrides DNS (iCloud Private Relay, a real VPN, or Split-Tunnel fail-safe)** takes precedence whenever it’s active\[1\]\[2\]. | Look-ups may go straight to the alternative resolver, *bypassing* the block-lists until the feature is off. | 
| AdGuard running in **Local-VPN** mode – default Split-Tunnel | If the chosen DNS server is slow, Apple’s Split-Tunnel safety switch can temporarily fall back to the network-assigned resolver to keep connectivity\[2\]. | During that fallback window ads/tracker domains resolve normally. | 
| AdGuard running in **Local-VPN** mode – Full-Tunnel | iOS will not fall back; if AdGuard’s resolver is down the device simply loses connectivity\[2\]. | No silent bypass, but potential “no Internet” moments. | 
| A separate VPN app is enabled | Only one VPN slot exists; enabling another VPN automatically tears down AdGuard’s local VPN. DNS then follows the other VPN’s settings\[1\]\[3\]. | Complete bypass of AdGuard unless it is in Native mode. | 

### 3\. Practical take-aways

- **Local-VPN modes give you the most consistent filtering,** but you must accept that:\
   – Any real VPN, iCloud Private Relay, or another “Personal VPN” app can’t run simultaneously.\
   – Split-Tunnel’s safety mechanism may leak look-ups if your chosen resolver becomes slow. Switch to *Full-Tunnel* if that matters more to you than occasional outages.

- **Native mode lets you keep Private Relay or a third-party VPN,** yet Apple is free to override the DNS profile whenever those services are active. Cosmetic Safari blocking continues via the content-blocker extension, but system-wide DNS filtering is suspended until the override ends.

- On macOS the same rules apply: AdGuard can operate as a system extension (local VPN) or via a configuration profile. Any network interface that runs its own tunnel (for example a corporate VPN) will bypass AdGuard’s tunnel and therefore its DNS filtering.

### 4\. Tips to minimise unintentional bypass

1. Use **Full-Tunnel** inside AdGuard if you do not need other VPN products.  

2. If you must keep Native mode, pick a resolver that supports **DNS-over-HTTPS/QUIC** so even when Apple sends queries directly, they remain encrypted.  

3. Disable “Limit IP Address Tracking” per Wi-Fi network when you rely on AdGuard DNS; that toggle silently invokes Private Relay for that network, hijacking DNS\[4\].  

4. Monitor AdGuard’s query log; sudden gaps or spikes often reveal a bypass event.

\[1\] Reddit discussion explaining that AdGuard Native DNS is ignored whenever another VPN is active.\
\[2\] AdGuard Knowledge-base: Split-Tunnel fall-back behaviour and Full-Tunnel guarantee.\
\[3\] Apple allows only one Personal-VPN tunnel; enabling a second tears down the first.\
\[4\] Apple Community thread: the “Limit IP Address Tracking” toggle causes DNS to bypass custom settings.

Sources
\[1\] Adguard Pro iOS - DNS Protection settings confusing - Reddit <https://www.reddit.com/r/Adguard/comments/1aoagv7/adguard_pro_ios_dns_protection_settings_confusing/>
\[2\] Low-level Settings guide | AdGuard Knowledge Base <https://adguard.com/kb/adguard-for-ios/solving-problems/low-level-settings/>
\[3\] How to use Mullvad VPN at same time as AdGuard DNS on iOS? <https://discuss.privacyguides.net/t/how-to-use-mullvad-vpn-at-same-time-as-adguard-dns-on-ios/28699>
\[4\] iOS AdGuard Safari Extension with third party VPN - GL.iNet Forum <https://forum.gl-inet.com/t/ios-adguard-safari-extension-with-third-party-vpn/35801>
\[5\] Connect to public AdGuard DNS server <https://adguard-dns.io/en/public-dns.html>
\[6\] NextDNS iOS/MacOS profile vs Adguard Pro native custom DNS ... <https://www.reddit.com/r/nextdns/comments/1345mk7/nextdns_iosmacos_profile_vs_adguard_pro_native/>
\[7\] The Detailed In-depth Review about DNS Bypass iCloud - FoneLab <https://www.fonelab.com/resource/icloud-dns-bypass.html>
\[8\] VPN: can't get WireGuard & AdGuard working - Cloudron Forum <https://forum.cloudron.io/topic/12982/vpn-can-t-get-wireguard-adguard-working>
\[9\] Where is private dns option in MacOS - Apple Support Communities <https://discussions.apple.com/thread/255131168>
\[10\] Block Apple iCloud Private Relay from bypassing DNSFilter <https://help.dnsfilter.com/hc/en-us/articles/14811352360083-Block-Apple-iCloud-Private-Relay-from-bypassing-DNSFilter>
\[11\] AdGuard Pro — Adblock for iOS | Overview <https://adguard.com/en/adguard-ios-pro/overview.html>
\[12\] Advanced Settings guide | AdGuard Knowledge Base <https://adguard.com/kb/adguard-for-mac/solving-problems/advanced-settings/>
\[13\] iOS bypasses the configured DNS servers w… - Apple Communities <https://discussions.apple.com/thread/255710430>
\[14\] DNS ad blockers without local VPN - Privacy Guides Community <https://discuss.privacyguides.net/t/dns-ad-blockers-without-local-vpn/17879>
\[15\] Ad blocker for Mac by AdGuard: remove all kinds of ads <https://adguard.com/en/adguard-mac/overview.html>
\[16\] iOS 14+: Apps Bypassing Pi-hole using Encrypted DNS - Reddit <https://www.reddit.com/r/pihole/comments/l69exz/ios_14_apps_bypassing_pihole_using_encrypted_dns/>
\[17\] LAN DNS resolution fails through OpenVPN when Adguard is enabled <https://forum.opnsense.org/index.php?topic=30757.0>
\[18\] Anyone here use Adguard? - Questions - Privacy Guides Community <https://discuss.privacyguides.net/t/anyone-here-use-adguard/28623>
\[19\] iCloud DNS Bypass | How to Bypass Activation Lock on iPhone \[2025\] <https://www.youtube.com/watch?v=97W_585vJ7o>
\[20\] Block ads while using VPN for privacy? - Pi-hole Userspace <https://discourse.pi-hole.net/t/block-ads-while-using-vpn-for-privacy/39657>

## Using AdGuard Pro and iCloud Private Relay together 

### Short answer

Yes, **they can coexist, but only if you use AdGuard Pro in its *DNS-only* mode (Native DNS profile)**. Any AdGuard mode that relies on the “local VPN” tunnel will disable or break iCloud Private Relay because both services need to own the single VPN slot that iOS allows\[1\]\[2\].

### Why full-VPN ad-blocking conflicts with Private Relay

1. iOS treats both a VPN app (including AdGuard’s “local VPN” filtering) and iCloud Private Relay as network extensions that require the sole system-wide VPN interface.  

2. When AdGuard runs in this mode it takes the slot first; Private Relay detects a competing tunnel and automatically switches itself off, or Safari traffic bypasses AdGuard and no filtering happens\[1\].

### How to run them side by side

| Step | Setting | Where to change | 
|---|---|---|
| 1 | Turn **off** AdGuard’s “Protection” toggle that says it will create a local VPN tunnel. | AdGuard Pro → Protection tab | 
| 2 | In AdGuard Pro, open **DNS Protection** → choose **Native (encrypted) DNS** and select an AdGuard DNS server or your custom block-list. This installs an iOS configuration profile and *does not* use the VPN slot. | AdGuard Pro | 
| 3 | Enable **iCloud Private Relay** (Settings → Apple ID → iCloud → Private Relay). | iOS Settings | 
| 4 | Optional: keep AdGuard’s Safari content blocker extensions on for cosmetic filtering inside Safari pages; these do not interfere with Private Relay. | Settings → Safari → Extensions | 

In this arrangement:  

- Safari traffic is still routed through Apple’s dual-hop relay, hiding your IP.  

- DNS lookups for all apps go to AdGuard’s resolver, so domain-level ad/tracker blocking still happens\[3\]\[2\].  

- System-wide TCP/UDP packet filtering (blocking in-app ads that load by IP address) is *not* possible because that requires the local VPN tunnel which Private Relay precludes.

### Limitations you should be aware of

- Cosmetic element hiding and advanced HTTPS filtering outside Safari won’t work, because those need the VPN tunnel.  

- AdGuard’s “Block Ads in Apps” switch has no effect in DNS-only mode.  

- The AdGuard test page may claim you’re “not using AdGuard DNS” when Private Relay is on; reload the page via Safari’s “Show IP Address” option to verify\[3\].  

- If you ever toggle AdGuard back to “local VPN,” iOS will instantly suspend Private Relay until you disable the VPN again.

### Bottom line

Run **AdGuard Pro in Native DNS mode** for lightweight system-wide ad/tracker blocking, keep **iCloud Private Relay enabled** for IP anonymization in Safari and Mail, and accept that deeper packet-level filtering is impossible while Apple occupies the VPN slot.

\[1\] AdGuard knowledge-base article explaining that Private Relay encrypts traffic before AdGuard can filter, so the two cannot work when AdGuard uses the VPN tunnel.\
\[3\] AdGuard DNS guide and community discussion describing success with Private Relay + Native DNS profile.\
\[2\] Reddit thread confirming that iOS allows custom DNS alongside Private Relay only when AdGuard avoids the VPN implementation.

Sources
\[1\] iCloud Private Relay and AdGuard <https://adguard.com/kb/adguard-for-mac/solving-problems/icloud-private-relay/>
\[2\] Using alongside iCloud Private Relay - AdGuard DNS <https://adguard-dns.com/kb/private-dns/solving-problems/icloud-private-relay/>
\[3\] What does it mean to use iCloud Private Relay along with Adguard ... <https://www.reddit.com/r/Adguard/comments/1741ign/what_does_it_mean_to_use_icloud_private_relay/>
\[4\] Compatibility issues with different macOS versions - AdGuard <https://adguard.com/kb/adguard-for-mac/solving-problems/big-sur-issues/>
\[5\] How to turn off iCloud Private Relay without an iCloud+ subscription? <https://discussions.apple.com/thread/253551794>
\[6\] How to Block iCloud Private Relay - Kolide <https://www.kolide.com/features/checks/mac-disable-icloud-private-relay>
\[7\] Protect Mail Activity and AdGuard <https://adguard.com/kb/adguard-for-mac/solving-problems/protect-mail-activity/>
\[8\] How to turn off iCloud Private Relay without an iCloud+ subscription? <https://discussions.apple.com/thread/253551794?page=2>
\[9\] Private Relay and Adguard safari extension - MacRumors Forums <https://forums.macrumors.com/threads/private-relay-and-adguard-safari-extension.2368629/>