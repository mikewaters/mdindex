---
tags:
  - document 📑
---
# Making Digital Payments Securely

Here I define “Secure” not as **anonymous**, but matching [Our Threat Model.md](./Our%20Threat%20Model.md). 

## Notes

- Prepaid cards generally require registration/KYC to use online, and so violates the threat model

- Some vendors accept cash in the mail, like Proton and Mullvad. This is a little freaky, I am not a criminal or anything.

# Research

## Partially anonymous methods and services

Here are several methods and services, that when used as described in this section, provide you with partial anonymity.

## 1\. Prepaid cards (gift cards)

![Pasted 2024-07-29-18-23-30.jpeg](./Making%20Digital%20Payments%20Securely-assets/Pasted%202024-07-29-18-23-30.jpeg)

Prepaid cards, also known as prepaid credit cards, prepaid debit cards, and prepaid gift cards, are **cards that you buy with a certain amount of money loaded onto them**. You can use them like a credit card or a debit card, with a few crucial differences.

- Prepaid cards are not credit cards. With a credit card, you are borrowing money every time you use the card, then paying off your debt later. With a prepaid card, you aren’t borrowing money. You are simply using the money that has already been loaded onto the card’s account.

- Prepaid cards are not debit cards. A debit card is linked to your bank account and draws money from that as necessary. Prepaid cards are not linked to your bank account. They simply draw money from their own account until the prepaid amount is used up.

You can buy prepaid cards in many places, including online or from a **physical store** (with cash). How you buy them, and what you do with them after you have used up all the money, determines whether a prepaid card is partially anonymous, or fully anonymous. Be careful, however, as the [FTC warns](https://consumer.ftc.gov/articles/avoiding-and-reporting-gift-card-scams)about some prepaid card payment scams.

If you simply want to conceal your identity from whoever you are transacting with, prepaid cards offer **enormous flexibility**. You can buy your prepaid cards in person at some store, or online with a credit card. It really doesn’t matter, because the person you are transacting with still won’t know who you are.

You may be wondering what to do with the little plastic card once you’ve used all the money in its account. In general, the card will go in the garbage once you use the money loaded onto it. Some prepaid cards can be reloaded anonymously with cash or perhaps with cryptocurrency. This is fine, assuming again that you don’t need to provide an ID to reload your card.

If a card allows you to reload it online, the card provider will get some information about you during the process. You’ll still be anonymous from the perspective of the person you are transacting with.

#### Note: Use a burner email!

This is probably a good time to mention that email address databases are also getting breached with shocking regularity. One potential solution is to use a temporary burner email address, which conceals and protects your regular email address, while still letting you purchase from and register on various websites.

We have an entire guide devoted to [temporary disposable email](https://restoreprivacy.com/email/temporary-disposable/) address that you should check out.

## 2\. Virtual credit cards

Virtual credit card services, such as [Privacy.com](Privacy.com), have both pros and cons. For one, these are not anonymous payment methods because:

- The virtual card service knows your identity (and is linked to your bank account or credit card)

- Every transaction can be seen by the virtual card service

Despite these drawbacks, however, there are still advantages to virtual card services. For example, here are some benefits to [privacy.com](privacy.com), which is a popular service in the United States:

- You can hide your identity from the merchant where you are purchasing goods or services.

- You can create burner cards that expire after a single transaction.

- You can set spending limits on cards, which prevents unauthorized payments.

- You can delete cards at any time, so they can no longer be used.

- Cards are locked to the first merchant they are used with, preventing unauthorized use by third parties.

![Pasted 2024-07-29-18-23-30 1.jpeg](./Making%20Digital%20Payments%20Securely-assets/Pasted%202024-07-29-18-23-30%201.jpeg)

I have been using [Privacy.com](Privacy.com) for a few years and find it to be a useful service when I want more privacy with transactions or more control over how the virtual card will be used by merchants. Being able to delete cards and set limits certainly has some advantages in a world with unauthorized charges.