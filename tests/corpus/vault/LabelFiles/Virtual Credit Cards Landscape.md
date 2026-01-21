# Virtual Credit Cards Landscape

## Virtual Credit-Card Services and the Privacy They Really Provide

### Fast take

Most “virtual” cards obscure your real card number, but they differ sharply in what they reveal to merchants, how treat your data, what they cost, and how much of your shopping trail reaches your bank.\
The privacy-first specialists ([Privacy.com](http://Privacy.com), Blur/IronVest) keep both the store *and* your bank largely in the dark, while bank-issued options (Capital One Eno, Citi VAN) give you convenience but almost no added data privacy against the issuer itself. Fin-tech wallets (Wise, Revolut) sit in the middle.

### Side-by-side comparison

| Provider | What the merchant learns about you | Provider’s data-sharing posture | Direct user cost | What your funding bank sees | Extra privacy notes | 
|---|---|---|---|---|---|
| **[Privacy.com](http://Privacy.com)** | Only the virtual card number and billing data you type; name optional\[1\]. | Policy states it **never sells or shares** customer data for marketing; uses data only for compliance and service\[2\]. | Basic tier free (3% FX fee)\[3\]; Pro $10/mo, Premium $25/mo\[4\]\[5\]. | Funding bank sees ACH debits labelled “PRIVACY\*xxxx”, **no merchant detail**. | Merchant-locked cards and single-use cards stop reuse; Plaid link means Plaid gets account meta-data\[6\]. | 
| **IronVest / Abine Blur (masked cards)** | A one-time card number, alias name and alias e-mail; balance hard-capped\[7\]\[8\]. | Explicit “**We never sell or share your data** without consent”\[9\]. | Unlimited masked cards in “Premium” $14.99/mo or $99/yr\[10\]. | Real card statement shows only a charge to **IronVest/Abine**, not the merchant. | Optional biometric unlock for every masked transaction\[11\]; Abine had a 2018 S3 exposure (no card data leaked)\[12\]. | 
| **Wise digital card** | Normal checkout details, but number can be frozen or replaced after each use\[13\]. | Does **not sell data**; shares with service providers & marketing partners, with opt-out rights\[14\]. | 1st three virtual cards free; physical card \~£7; FX from 0.43%\[13\]. | Bank sees only top-ups to Wise, never the end merchant. | Up to 3 live cards; supports 40+ currencies; card lives only in app (less risk if phone lost)\[15\]. | 
| **Revolut disposable / multi-use card** | Multi-use card = normal recurring use; disposable card number dies after one purchase\[16\]\[17\]. | Shares data with credit bureaus, analytics and ad partners by default (opt-out available)\[18\]\[19\]. | Virtual cards free; top-ups by non-US debit  ≤ 3%; paid plans from $10/mo\[20\]\[21\]. | Funding bank sees generic “REVOLUT\*Top-up” entries; no merchant data. | Up to 20 virtual cards + 1 disposable; easy freeze/unfreeze; company is a frequent marketing user of spend data\[18\]. | 
| **Capital One Eno** | Merchant-specific card number; store still sees your legal name and address\[22\]. | Bank privacy notice allows sharing of purchase data with affiliates for marketing, cannot be limited\[23\]. | Free for Capital One cardholders; normal card APR & FX apply\[22\]\[24\]. | Issuer *is* the bank, so it has the full merchant detail by design. | Locked cards may still accept “recurring” charges even when frozen\[25\]. | 
| **Citi Virtual Account Number (VAN)** | Virtual number and billing info; you supply name/address\[26\]. | Citi shares data with affiliates for everyday business and marketing by default; limited opt-outs\[27\]\[28\]. | Free for eligible Citi cards\[26\]; underlying card fees apply. | Issuer sees entire purchase information. | Spending limits can fail on some recurring charges\[29\]; only certain Citi cards supported\[30\]. | 

### What the classifications mean

1. **Privacy from vendors**  

   - Highest: masked/disposable numbers that cannot be reused and do not contain your real name ([Privacy.com](http://Privacy.com) burner, Revolut disposable, IronVest).  

   - Medium: merchant-locked virtual numbers ([Privacy.com](http://Privacy.com) merchant card, Eno, Citi VAN).  

   - Low: tokenised wallets (Apple Pay, Google Pay) are secure, but the merchant still gets your real name and billing address.  

2. **Provider’s privacy policy**  

   - *No selling / minimal sharing*: [Privacy.com](http://Privacy.com)\[2\], IronVest\[9\], Wise\[14\].  

   - *Marketing sharing opt-out*: Revolut (default opt-in)\[18\], Wise (can object)\[14\].  

   - *Broad affiliate sharing*: Capital One\[23\], Citi\[27\].  

3. **Cost of use**  

   - Truly free (beyond underlying card FX): Capital One Eno, Citi VAN.  

   - Free tier with limits: [Privacy.com](http://Privacy.com), Wise, Revolut.  

   - Subscription required for unlimited masked cards: IronVest Blur.  

4. **Privacy with the origin bank**  

   - If the *card issuer is also the bank* (Capital One, Citi), the bank inevitably sees the merchant.  

   - For *overlay* services ([Privacy.com](http://Privacy.com), Revolut, Wise, IronVest) the funding institution sees only a top-up or ACH pull and **never gets merchant-level data** — useful when you prefer your primary bank not to know what you bought.  

### Other noteworthy implications

- **Data-broker exposure** – Revolut announced automatic data feeds to credit bureaus and social-media analytics in 2019; users must manually opt out\[18\].  

- **Biometric locks** – IronVest lets you require a face-match for each masked-card checkout, preventing use even if the virtual number leaks\[11\].  

- **Chargeback handling** – Bank-issued virtual numbers (Eno, Citi) follow normal credit-card dispute rules; [Privacy.com](http://Privacy.com) and IronVest handle disputes internally and refund first, then pursue the merchant, reducing headache for the buyer\[2\]\[31\].  

- **Recurring-charge leaks** – Both Citi VAN and Capital One Eno can let ongoing subscription charges hit even after you “lock” the virtual card if the merchant flags the transaction as recurring\[25\]\[29\]. Deleting the card (not just locking) is safer.  

- **Foreign-currency pricing** – Wise and Revolut convert at interbank or near-interbank rates with an upfront fee (0.43% Wise\[13\], 0.4-1% Revolut\[16\]); bank virtual cards inherit the usually poorer FX terms of the underlying credit card.  

- **Regulatory footprint** – [Privacy.com](http://Privacy.com) cards are issued by Patriot Bank N.A. and regulated like charge cards, so there’s no credit pull\[32\]; Wise and Revolut operate under e-money licences in the EEA/UK and are not covered by US Regulation E on error resolution outside their U.S. subsidiary accounts.  

### Choosing the right option

- **Maximum anonymity from both store and bank** → [Privacy.com](http://Privacy.com) burner cards or IronVest masked cards.  

- **No-cost convenience for an existing credit-card account** → Capital One Eno or Citi VAN (accept lower privacy vis-à-vis the issuer).  

- **Frequent international spending** → Wise virtual card (low FX) or Revolut disposable card if you already use Revolut.  

- **Single dashboard for passwords + email/phone masking** → IronVest Blur (bundles masked cards with broader identity shielding).  

Understanding *where* each data trail stops — at the merchant, the card provider, or your primary bank — lets you match a virtual-card service to your personal threat model and budget.

Sources
\[1\] What information does Privacy share with the merchant? <https://support.privacy.com/hc/en-us/articles/24450438423063-What-information-does-Privacy-share-with-the-merchant>
\[2\] Privacy Policy | Privacy Virtual Cards and Services <https://www.privacy.com/privacy-policy>
\[3\] How much does it cost to use Privacy? <https://support.privacy.com/hc/en-us/articles/360012046114-How-much-does-it-cost-to-use-Privacy>
\[4\] What is Privacy Pro? <https://support.privacy.com/hc/en-us/articles/360039640953-What-is-Privacy-Pro>
\[5\] What is Privacy Premium? <https://support.privacy.com/hc/en-us/articles/360039641013-What-is-Privacy-Premium>
\[6\] Has [privacy.com](http://privacy.com) gone too far? : r/PrivacySecurityOSINT - Reddit <https://www.reddit.com/r/PrivacySecurityOSINT/comments/17ijcnw/has_privacycom_gone_too_far/>
\[7\] Blur from Privacy Company Abine Keeps Your Personal Financial Data Safe by Blocking an Average of 175,000 Online Trackers Annually <https://www.cardrates.com/news/blur-from-privacy-company-abine-keeps-personal-financial-data-safe/>
\[8\] How To Use A 'Fake' Credit Card To Protect Yourself From Hackers <https://www.businessinsider.com/abine-maskme-protects-against-hackers-2015-1>
\[9\] IronVest's privacy policy <https://ironvest.com/legal/privacy-policy/>
\[10\] Abine Blur Review (2025): Password Manager Safety at a Cost <https://vpnoverview.com/password-managers/blur-review/>
\[11\] How to Enable Biometric Protection for Your Masked Cards - Groove <https://ironvest-kb.groovehq.com/help/how-to-enable-biometric-protection-for-your-masked-cards>
\[12\] Data on 2.4M Abine Blur Users 'Potentially Exposed' <https://www.pcmag.com/news/data-on-24m-abine-blur-users-potentially-exposed>
\[13\] Virtual Card | Create your Wise Virtual Debit Card - Wise <https://wise.com/au/virtual-card/>
\[14\] Personal Customer Privacy Notice - Wise <https://wise.com/gb/legal/privacy-notice-personal-en>
\[15\] Virtual Card | Create your Wise Virtual Debit Card <https://wise.com/nz/virtual-card/>
\[16\] Virtual card | Create and spend with a virtual debit card online <https://www.revolut.com/cards/virtual-card/>
\[17\] Shop online with disposable virtual cards for extra security - Revolut <https://www.revolut.com/en-JP/blog/post/disposable-virtual-cards-en/>
\[18\] Why Revolut is going to start sharing your data with credit bureaus <https://www.siliconrepublic.com/enterprise/revolut-privacy-policy-credit-bureaus-targeted-advertising>
\[19\] Customer Privacy Notice | Revolut United Kingdom <https://www.revolut.com/legal/privacy/>
\[20\] Premium Fees <https://assets.revolut.com/terms_and_conditions/pdf/technologies_inc_Personal_Fees_(Premium)_0.0.3_2021-07-23_en.pdf>
\[21\] Personal Fees | Revolut United States <https://www.revolut.com/en-US/legal/standard-fees/>
\[22\] Virtual Card Numbers from Eno FAQ | Capital One <https://www.capitalone.com/digital/tools/eno/virtual-card-numbers/help/>
\[23\] Capital One Online Privacy Policy <https://www.capitalone.com/privacy/online-privacy-policy/>
\[24\] What Is a Virtual Credit Card Number? | Capital One <https://www.capitalone.com/learn-grow/money-management/what-are-virtual-card-numbers/>
\[25\] Capital One Eno - Charge made to locked virtual number - Reddit <https://www.reddit.com/r/CreditCards/comments/j34i08/capital_one_eno_charge_made_to_locked_virtual/>
\[26\] Citi Virtual Credit Card—What It Is and How To Get One <https://www.privacy.com/blog/citi-virtual-credit-card>
\[27\] WHAT DOES CITI DO WITH YOUR PERSONAL INFORMATION? <https://online.citi.com/JRS/popups/Annual_Privacy_Notice.pdf>
\[28\] Rev. November 2019 <https://online.citi.com/JRS/popups/PrivacyNoticeEN.pdf>
\[29\] Citi Virtual Account Number "Limits" aren't really Limits : r/CreditCards <https://www.reddit.com/r/CreditCards/comments/1c98ayd/citi_virtual_account_number_limits_arent_really/>
\[30\] FAQ and Update: Citi Virtual Account Numbers : r/CreditCards - Reddit <https://www.reddit.com/r/CreditCards/comments/lbuolu/faq_and_update_citi_virtual_account_numbers/>
\[31\] Payment Protection with IronVest - A [Privacy.com](http://Privacy.com) Alternative <https://ironvest.com/blog/ironvest-a-privacy.com-alternative-for-better-card-and-bank-account-protection/>
\[32\] Privacy Virtual Cards <https://www.privacy.com>
\[33\] Cardholder Agreement | Getting Started with Privacy Virtual Cards <https://privacy.com/cardholder-agreement>
\[34\] Is [Privacy.com](http://Privacy.com) Safe? - JoinDeleteMe <https://joindeleteme.com/is-site-safe/is-privacy-com-safe/>
\[35\] [Privacy.com](http://Privacy.com) virtual debit card. Do my purchases show the company ... <https://www.reddit.com/r/privacy/comments/g4dorf/privacycom_virtual_debit_card_do_my_purchases/>
\[36\] Privacy Policy for Privacy Agreement - [PrivacyPolicies.com](http://PrivacyPolicies.com) <https://www.privacypolicies.com/live/3cae4ddf-c2fc-4663-b49c-80d8578d7b0a>
\[37\] Cardholder Agreement | Getting Started with Privacy Virtual Cards <https://www.privacy.com/cardholder-agreement>
\[38\] Virtual Cards To Protect Your Payments <https://www.privacy.com/virtual-card>
\[39\] [Privacy.com](http://Privacy.com) <https://tosdr.org/en/service/3262>
\[40\] [Privacy.com](http://Privacy.com) rebrands to Lithic, raises $43M for virtual payment cards | TechCrunch <https://techcrunch.com/2021/05/20/privacy-com-rebrands-to-lithic-raises-43m-for-virtual-payment-cards/?guccounter=2>
\[41\] Is [Privacy.com](http://Privacy.com) Safe? An Overview of Privacy's Security Practices <https://www.privacy.com/blog/is-privacy-com-safe-an-overview-of-privacys-security-practices>
\[42\] Terms and Conditions | Privacy Virtual Cards and Services <https://www.privacy.com/terms>
\[43\] Privacy's Paid Subscription Plans <https://www.privacy.com/blog/subscription-plan>
\[44\] Privacy - Overview, News & Similar companies | [ZoomInfo.com](http://ZoomInfo.com) <https://www.zoominfo.com/c/privacy-inc/410762954>
\[45\] Vulnerability Disclosure Policy - Privacy Virtual Cards <https://www.privacy.com/legal/security>
\[46\] Exploring the Capital One Virtual Credit Card <https://www.privacy.com/blog/capital-one-virtual-credit-card>
\[47\] How Eno Can Help You Manage Finances | Capital One <https://www.capitalone.com/learn-grow/money-management/eno-manages-finances/>
\[48\] How Capital One wants to make online shopping safer <https://digiday.com/marketing/capital-one-wants-make-online-shopping-safer/>
\[49\] Set Up Virtual card Numbers with Eno | Capital One <https://www.capitalone.com/digital/tools/eno/virtual-card-numbers/setup/>
\[50\] Eno® from Capital One® - Chrome Web Store [https://chromewebstore.google.com/detail/eno®-from-capital-one®/clmkdohmabikagpnhjmgacbclihgmdje?hl=en](https://chromewebstore.google.com/detail/eno%C2%AE-from-capital-one%C2%AE/clmkdohmabikagpnhjmgacbclihgmdje?hl=en)
\[51\] Capital One Shopping, a free tool to find deals online <https://www.capitalone.com/learn-grow/money-management/capital-one-shopping/>
\[52\] A Unique Virtual Card Number from Eno | Capital One <https://www.capitalone.com/digital/tools/eno/virtual-card-numbers/congratulations/>
\[53\] What happens when a Capital One virtual card gets frauded? - Reddit <https://www.reddit.com/r/CreditCards/comments/182dded/what_happens_when_a_capital_one_virtual_card_gets/>
\[54\] Capital One Eno virtual assistant: Things to know | [CreditCards.com](http://CreditCards.com) <https://www.creditcards.com/card-advice/capital-one-eno/>
\[55\] Eno® from Capital One® [https://chrome.google.com/webstore/detail/eno®-from-capital-one®/clmkdohmabikagpnhjmgacbclihgmdje/related](https://chrome.google.com/webstore/detail/eno%C2%AE-from-capital-one%C2%AE/clmkdohmabikagpnhjmgacbclihgmdje/related)
\[56\] Eno Card Protection & Security | Benefits - Capital One <https://www.capitalone.com/digital/tools/eno/card-security/>
\[57\] Help | Chat With Eno: Text or Talk Online | Capital One <https://www.capitalone.com/digital/tools/eno/ask-eno/>
\[58\] Capital One Virtual Credit Card Numbers Allow for Safer Online Shopping <https://www.mybanktracker.com/credit-cards/faq/capital-one-virtual-credit-card-numbers-297284>
\[59\] Virtual Credit Cards for Online Shopping - Capital One <https://www.capitalone.com/learn-grow/money-management/virtual-cards-shopping-online/>
\[60\] Get Help With Spending Insights From Eno | Capital One <https://www.capitalone.com/digital/tools/eno/spending-insights/>
\[61\] How to Use a Virtual Card | Capital One <https://www.capitalone.com/learn-grow/money-management/how-to-use-virtual-cards-from-capital-one/>
\[62\] ICG_Pres_CTS (Letter) <https://www.citi.com/tts/sa/texas_mn/assets/docs/VCAfinal.pdf>
\[63\] \[PDF\] Citi® Virtual Card Accounts <https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/case-studies/1135896_GTS25543_VirtualCardAcct_SS_vF_27Sept2013.pdf>
\[64\] FACTS <http://www.citibank.com/citigoldprivateclient/homepage/pdfs/CGMIPrivacyNoticeINAI_E.pdf>
\[65\] Citi \[Virtual Account Number\] question - [Bogleheads.org](http://Bogleheads.org) <https://www.bogleheads.org/forum/viewtopic.php?t=369697>
\[66\] Virtual Account Numbers - Citi® Card Benefits <https://www.cardbenefits.citi.com/Products/Virtual-Account-Numbers>
\[67\] \[PDF\] Citibank Privcy Notice <https://online.citi.com/JRS/popups/ao/Citibank_Consumer_Privacy_Notice.pdf>
\[68\] \[PDF\] Citi® Virtual Card Accounts for Travel <https://www.citigroup.com/tts/solutions/commercial-cards/assets/docs/cards/virtual-card-accounts.pdf>
\[69\] Select language <https://www.proquest.com/docview/198798554>
\[70\] \[PDF\] PRIVACY STATEMENT - [Citi.com](http://Citi.com) <https://www.citibank.com/icg/sa/emea/hungary/english/assets/docs/CITI_HOLDINGS_Hungary.pdf>
\[71\] Citi: Using Citi's Virtual Account Number (VAN) <https://www.youtube.com/watch?v=mdrpPODCG50>
\[72\] The Privacy Policy (the "Policy") applies only to institutional banking products or services of <https://www.citi.com/tts/sa/tts-privacy-statements/assets/docs/CitiDirect_Privacy_Terms_and_Condition_China.pdf>
\[73\] What You Should Know About Citibank Virtual Credit Cards <https://www.gobankingrates.com/credit-cards/advice/what-should-know-citibank-virtual-credit-cards/>
\[74\] CITIBANK NA (UAE), CITIBANK NA (DIFC) AND <https://www.citigroup.com/tts/sa/tts-privacy-statements/assets/docs/CIT-Privacy-Notice-Customers.pdf>
\[75\] Transaction Services <https://www.citi.com/tts/solutions/commercial-cards/assets/docs/case-studies/1078311_GTS25823_VCA_Custom_Integrated_Solutions_SS.pdf>
\[76\] Citi Personal <https://online.citi.com/JRS/popups/CPWM_Privacy_Notice.pdf>
\[77\] Can a disposable card be tracked? : r/Revolut - Reddit <https://www.reddit.com/r/Revolut/comments/v35gwj/can_a_disposable_card_be_tracked/>
\[78\] Business fees (Scale) <https://assets.revolut.com/terms_and_conditions/pdf/technologies_inc_Business_fees_(Scale)_0.0.2_2021-07-23_en.pdf>
\[79\] Prospective customer privacy notice | Revolut United Kingdom <https://help.revolut.com/business/help/more/general/what-personal-data-does-revolut-business-process-about-me-as-a-prospect/>
\[80\] Revolut Virtual Card Pros & Cons | DoNotPay <https://donotpay.com/learn/revolut-virtual-card/>
\[81\] Revolut Business Customer Privacy Notice <https://www.revolut.com/en-LT/legal/business-customer-privacy-notice/>
\[82\] Create and spend with a virtual prepaid debit card - Revolut <https://www.revolut.com/en-US/cards/virtual-card/>
\[83\] Is the card free(virtual) : r/Revolut - Reddit <https://www.reddit.com/r/Revolut/comments/1llqedd/is_the_card_freevirtual/>
\[84\] \[PDF\] Privacy Policy - Revolut <https://assets.revolut.com/terms_and_conditions/pdf/payments_uab_Privacy_Policy_0.1.0_1643212203_en.pdf>
\[85\] How to use Revolut virtual card? | Revolut Germany <https://help.revolut.com/en-DE/help/cards/get-a-card/how-to-use-virtual-cards/>
\[86\] Complete guide about Revolut virtual cards in the US - Wise <https://wise.com/us/blog/revolut-virtual-card>
\[87\] Customer Privacy Notice | Revolut Liechtenstein <https://www.revolut.com/en-LI/legal/privacy/>
\[88\] How to use Revolut virtual card? | Revolut Poland <https://help.revolut.com/en-PL/help/cards/get-a-card/how-to-use-virtual-cards/>
\[89\] Multi-Currency Card | Revolut United States <https://www.revolut.com/en-US/cards/>