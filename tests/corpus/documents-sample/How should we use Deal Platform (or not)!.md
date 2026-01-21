# How should we use Deal Platform (or not)?

\#dealxg #dealapi

tl;dr: If we did not originate the deal, we should not write our deal changes to the Deal Platform. If we did originate the deal, we should write it to the Deal Platform.



- What is the distinction between a Platform and a Database? Where does DealXG appear to sit, and what are the risks?

## The model that I prefer, for Finance and for Risk

uniFI has several transactions of interest here, which can be lumped into a few buckets which can be independently managed:

1. Financing iteration

   1. *May significantly change the deal structure*

   2. Buyer risk assessment

   3. Loan acquisition

2. Regulatory compliance

   1. *Should not modify a deal*

3. Deal initiation and ingestion

   1. *Creates a new deal within FR, with varied goals:*

      1. *Complete within unIFI*

      2. *Structure outside of uniFI (R360) and complete within FR*

      3. *Structure and complete outside of FR*

   2. Consumer credit

      1. UCA and eBiz

   3. Retailer credit

      1. Finance Services and Finance Driver

   4. Lender deal ingestion

      1. Lender referral

      2. External decision

   5. Legacy data pipelines

      1. Lead ingestion

      2. Deal ingestion

         1. DMS integrations (etc)


