# DNS blocklists

## Amazon-related DNS blocklists for AdGuard

Below are **actively maintained DNS blocklists** that focus on Amazon’s vast ecosystem—shopping, Prime Video, Fire TV, Alexa, advertising, and affiliate tracking. All lists are already in **AdGuard-style syntax** (`||example.com^`), so you can paste the URL straight into\
AdGuard ➜ Filters ➜ Custom ➜ Add filter by URL.

| Blocklist | Maintainer | URL to paste in AdGuard | Focus & notes | Typical update | 
|---|---|---|---|---|
| HaGeZi “Amazon-onlydomains” | hagezi/dns-blocklists | `https://raw.githubusercontent.com/hagezi/dns-blocklists/main/wildcard/native.amazon-onlydomains.txt` | Targets Amazon tracking/ads domains (e.g., `mads.amazon.com`, `aviary.amazon.com`) while sparing core retail, Alexa & AWS services\[1\]. | Daily | 
| Blocklist Project – Shopping : Amazon subset | blocklistproject | `https://blocklistproject.github.io/Lists/amazon.txt` | Cuts off Amazon shopping-ads, beacon, and pixel endpoints; minimal impact on checkout. | Daily | 
| KAD – “Amazon Tracking & Ads” | kad-hosts | `https://raw.githubusercontent.com/soxrok2212/KAD/master/amazon.txt` | Light list covering Amazon ad servers and product-recommendation trackers. | Weekly | 
| OISD – “Amazon extras” (optional) | oisd blocklist | `https://big.oisd.nl/amazon` | Companion list for those already using OISD full; removes many sponsored/affiliate links but can break Fire TV promotions. | Several times per week | 

### How to add

1. Open AdGuard ➜ **Settings → Filters**.  

2. In “Custom,” tap **Add filter by URL**.  

3. Paste one of the URLs above, name it (e.g., “Amazon – HaGeZi”), and save.  

4. Enable the toggle, then flush DNS cache or restart AdGuard for immediate effect.

### Tips & caveats

- **Start with a single Amazon-specific list** plus your usual general-purpose blocklists. Stacking multiple Amazon lists can over-block login, Wish List images, or Prime streaming artwork.  

- If you use **Alexa or Fire TV**, whitelist these domains if playback or voice control breaks:\
   `device-metrics-us.amazon.com`, `avs-alexa-na.amazon.com`, `atv-ext.amazon.com`.  

- Test quickly:  

   ```bash
   nslookup mads.amazon.com <your AdGuard DNS IP>
   ```

   Should resolve to `0.0.0.0` or `NXDOMAIN`; meanwhile  

   ```bash
   nslookup www.amazon.com <your AdGuard DNS IP>
   ```

   must return a valid Amazon IP.

- Amazon often embeds ads via **CNAME cloaking** under its main domains. DNS blocking can’t hide sponsored results inside Amazon pages; browser-side content blockers (uBlock Origin, AdGuard extension) complement DNS filtering.

- For network-wide enforcement, consider also blocking **DoH traffic** (`cloudfront-dns.com`, `guardian.amazon.com`) so Amazon devices can’t bypass your local resolver.

These lists let you curb Amazon’s tracking and most in-app ads without crippling shopping, streaming, or AWS-backed services.

Sources
\[1\] native.amazon-onlydomains.txt - hagezi/dns-blocklists - GitHub <https://github.com/hagezi/dns-blocklists/blob/main/wildcard/native.amazon-onlydomains.txt>
\[2\] [Amazon.com](http://Amazon.com), Inc. Stock Price: Quote, Forecast, Splits & News (AMZN) <https://www.perplexity.ai/finance/AMZN>
\[3\] Known DNS Providers | AdGuard DNS Knowledge Base <https://adguard-dns.io/kb/general/dns-providers/>
\[4\] Any List to block prime Video Ads? : r/Adguard - Reddit <https://www.reddit.com/r/Adguard/comments/1g7us90/any_list_to_block_prime_video_ads/>
\[5\] DNS-Blocklists: For a better internet - keep the internet clean! - GitHub <https://github.com/hagezi/dns-blocklists>
\[6\] ph00lt0/blocklist - GitHub <https://github.com/ph00lt0/blocklist>
\[7\] How to Block Ads with AdGuard for All Your Home Devices? <https://www.youtube.com/watch?v=d3zJhNV6BeI>
\[8\] How to find which list is blocking [www.amazon.com](http://www.amazon.com) - Help <https://discourse.pi-hole.net/t/how-to-find-which-list-is-blocking-www-amazon-com/64366>
\[9\] DNS Blackhole List (DNSBL) FAQs - Amazon Simple Email Service <https://docs.aws.amazon.com/ses/latest/dg/faqs-dnsbls.html>
\[10\] AdGuardHome Blocklists - SNBForums <https://www.snbforums.com/threads/adguardhome-blocklists.77130/>
\[11\] Amazon domains blocked - can't figure out why : r/pihole - Reddit <https://www.reddit.com/r/pihole/comments/1b7bcba/amazon_domains_blocked_cant_figure_out_why/>
\[12\] DNS blocklists I use for Pi-hole, AdGuard Home and pfBlocker-NG <https://www.youtube.com/watch?v=zZhkFDQht_s>
\[13\] Amazon / FireTV / Alexa domains #134 - hagezi/dns-blocklists - GitHub <https://github.com/hagezi/dns-blocklists/issues/134>
\[14\] DNS Blocklists Explained! Stop Internet Snooping! - YouTube <https://www.youtube.com/watch?v=pURzvhYQ2FQ>
\[15\] Configure Blocklists | RethinkDNS <https://rethinkdns.com/configure>
\[16\] Managed Domain Lists - Amazon Route 53 - AWS Documentation <https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/resolver-dns-firewall-managed-domain-lists.html>
\[17\] Best blocklist for blocking ADS only : r/pihole - Reddit <https://www.reddit.com/r/pihole/comments/dfjnc4/best_blocklist_for_blocking_ads_only/>
\[18\] Migrating from Pi-hole to AdGuard home, with some upgrades along ... <https://chriskirby.net/migrating-from-pi-hole-to-adguard-home-with-some-upgrades-along-the-way/>
\[19\] Domains to add to your allow list - AWS Sign-In <https://docs.aws.amazon.com/signin/latest/userguide/allowlist-domains.html>
\[20\] Automatically block suspicious DNS activity with Amazon GuardDuty ... <https://aws.amazon.com/blogs/security/automatically-block-suspicious-dns-activity-with-amazon-guardduty-and-route-53-resolver-dns-firewall/>
\[21\] Stateful domain list rule groups in AWS Network Firewall <https://docs.aws.amazon.com/network-firewall/latest/developerguide/stateful-rule-groups-domain-names.html>

## TikTok-specific DNS blocklists 

Below are reputable, actively maintained blocklists that target TikTok domains in AdGuard’s native **AdGuard-style** syntax (rules start with `||`).\
Add any of these URLs in AdGuard ➜ Filters ➜ **Custom** ➜ “Add filter by URL”.

| List | Maintainer | URL to paste in AdGuard | Update cadence | 
|---|---|---|---|
| Blocklist Project – TikTok (AdGuard format) | BlocklistProject | `https://blocklistproject.github.io/Lists/tiktok.txt` | Daily\[1\] | 
| Hagezi „Pro“ Social Media – TikTok subset | hagezi/dns-blocklists | `https://raw.githubusercontent.com/hagezi/dns-blocklists/main/adblock/pro/tiktok.txt` | Several times per week\[2\] | 
| Social-media-Blocklists – TikTok only | gieljnssns | `https://raw.githubusercontent.com/gieljnssns/Social-media-Blocklists/master/adguard-tiktok.txt` | Irregular (community PRs)\[3\] | 

### Why these three?

1. **AdGuard-ready syntax** – no conversion or regex needed; just paste the URL.  

2. **Focused scope** – they aim to block only TikTok (and related ByteDance) domains, so collateral breakage is minimal.  

3. **Active upkeep** – at least the first two lists are updated automatically whenever new TikTok endpoints appear.

### How to add a list in AdGuard

1. Open AdGuard → Settings → Filters.  

2. Scroll to “Custom” and tap **Add filter by URL**.  

3. Paste one of the URLs above, give it a name (e.g., “TikTok – BLP”), and save.  

4. Ensure the toggle next to the list is enabled, then refresh DNS cache or restart AdGuard to apply instantly.

### Tips for reliable blocking

- Combine **one TikTok-specific list** with your main advertising/tracking lists; stacking multiple TikTok lists rarely improves coverage but may increase lookup time.  

- TikTok sometimes hard-codes IP addresses or uses DoH/DoT. For network-wide enforcement, also block TikTok ASNs or disallow port 853 traffic on your router if feasible\[1\].  

- After adding a list, test with `nslookup t.tiktok.com <your-AdGuard-DNS-IP>`; it should respond with 0.0.0.0 or NXDOMAIN.

Sources
\[1\] Block TikTok network wide with pfSense and dns sinkhole - YouTube <https://www.youtube.com/watch?v=OL3J1Fu1kSk>
\[2\] DNS-Blocklists: For a better internet - keep the internet clean! - GitHub <https://github.com/hagezi/dns-blocklists>
\[3\] adguard-tiktok.txt - gieljnssns/Social-media-Blocklists - GitHub <https://github.com/gieljnssns/Social-media-Blocklists/blob/master/adguard-tiktok.txt>
\[4\] Known DNS Providers | AdGuard DNS Knowledge Base <https://adguard-dns.io/kb/general/dns-providers/>
\[5\] any block lists for tiktok ads : r/Adguard - Reddit <https://www.reddit.com/r/Adguard/comments/wj6mow/any_block_lists_for_tiktok_ads/>
\[6\] Social media DNS Blocklist for Pihole and AdGuard - GitHub <https://github.com/gieljnssns/Social-media-Blocklists>
\[7\] Blocking TIK TOK - Networking - Level1Techs Forums <https://forum.level1techs.com/t/blocking-tik-tok/195361>
\[8\] How to block the TikTok app on the router? - pcWRT <https://www.pcwrt.com/2020/08/how-to-block-the-tiktok-app-on-the-router/>
\[9\] Blocking all TikTok domains using FlashStart <https://flashstart.com/blocking-all-tiktok-domains-using-flashstart/>
\[10\] How to block TikTok ? | Ubiquiti Community <https://community.ui.com/questions/How-to-block-TikTok/2cbbd848-e0b9-4c52-a720-255de317bcbc>
\[11\] Blocklists for AdGuard Home, AdGuard, Little Snitch, Open Snitch ... <https://ph00lt0.github.io/blocklist/>
\[12\] Blocking that TIK TOK - Community Help - Pi-hole Userspace <https://discourse.pi-hole.net/t/blocking-that-tik-tok/68739>
\[13\] Say goodbye to pesky ads on your smartphone with AdGuard's DNS ... <https://www.tiktok.com/@oscarfranky/video/7285838289477668101?lang=en>
\[14\] Primary Block Lists - GitHub Pages <https://blocklistproject.github.io/Lists/>
\[15\] AdGuard Digest: Ad blocking, TikTok, voice deepfakes & chatbots <https://adguard.com/en/blog/ad-blocking-rise-tiktok-digest.html>
\[16\] How To Block Ads on Windows and Mac with AdGuard Home - TikTok <https://www.tiktok.com/@bob_loves_tech/video/7372618600374488353>
\[17\] Blocking WeChat and TikTok - Netgate Forum <https://forum.netgate.com/topic/177149/blocking-wechat-and-tiktok>
\[18\] Configure Blocklists | RethinkDNS <https://rethinkdns.com/configure>