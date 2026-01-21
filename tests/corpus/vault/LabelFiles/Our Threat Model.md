---
tags:
  - document 📑
---
# Our Threat Model

Is a:: Document

A good example of a real life threat model:

[X41-Mullvad-Audit-Public-Report-2024-12-10.pdf](../Card%20Library/X41-Mullvad-Audit-Public-Report-2024-12-10.pdf)

Want multiple providers for security sensitive needs that enable pseudo anonymity with a strong partition between them, such that no party (excepting LEA and government) can associate them.

Primarily we want to avoid IRL information from becoming part of our [Permanent Data Record.md](./Permanent%20Data%20Record.md), and so we need to avoid permitting IRL data in one system from being correlated in another system, and the easiest way to do this is by never tying any data point to your IRL identity (but this is not easy and sometimes not possible at all).

## Vendors/Providers

For each entity in a transaction, we should establish a risk profile. This should include: 

- What does the party know about your identity? 

- What ongoing data points of yours does the party have access to? 

- What could the party do with that data, and what do they claim to do? 

- What is the importance and sensitivity of the data, and what may happen if it is leaked? 

- [Example](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/761dcc06-188b-45a1-9cb0-50598dea1ffe#10ed9d39-7a7d-40cb-bf5c-a665742d75db)

## Digital Purchases

Given the deep data integrations used by the [Advertising Industrial Complex.md](./Advertising%20Industrial%20Complex.md), if one provider can identify a purchase you’ve made then it is possible or even likely that other providers can know this, and that data can become part of your [Permanent Data Record.md](./Permanent%20Data%20Record.md). For some purchases, you do not want this.

In order to avoid this correlation, we need to make decisions based on the relative security of the entities involved in a purchase: the Retailer, your Bank, and everything in between including ad networks linked to both ends of the transaction.

Example transaction: Buying privacy services.

- Product: Encrypted email

- Retailer: ProtonMail

   - Risk: I do not want the vendor to know my IRL identity at all, so I can not worry about how I use its services.

- Bank: Chase

   - Risk: Chase advertising partners may have access to purchase information, and this may include the product. Chase has access to a vast amount of my IRL data and identity going back decades.

   - Chase sells our transaction history [Chase Bank Tracking.md](./Chase%20Bank%20Tracking.md) and so we must obscure the Retailer from Chase (as it is clear from the name what they sell)