# Interop Notes

Apr 9, 2025

**Next actions (Product):**

- How should this be configured, and how do we ensure that uniFI and FS have access to the config?

   - [F237225](https://rally1.rallydev.com/#/?detail=/portfolioitem/feature/822943410589&fdp=true): Support configuring a dealer to skip conditional new dealjacket creation logic for Finance API and uniFI users

      - And then fill out each feature that needs it

- What are the change criteria?

   - What deal changes are allowed

   - When are deal changes allowed, precisely)

   - [F237449](https://rally1.rallydev.com/#/?detail=/portfolioitem/feature/823098579137&fdp=true): Spike: Define Permitted Unsubmitted Deal Changes

      - And then fill out each feature that needs it

- Does Finance Services need to validate synchronously? Or can it send events back?

- Does this need to be highly available? If not, we can defer the validation work.

- Discuss use cases where there are multiple deals, like:

   - Deal 1 is Ross, with bureau/IDV and submitted: declined

   - Deal 2 is Ross and Mike as coapp, unsubmitted etc

   - API client changes Mike to Michael

**Next actions (Arch, Eng):**

- Figure out validation architecture, if we need to:

   - [F237450](https://rally1.rallydev.com/#/?detail=/portfolioitem/feature/823098586043&fdp=true): Arch Spike: Define Validation Architecture

   - And then update the dependant feature: [F237227](https://rally1.rallydev.com/#/?detail=/portfolioitem/feature/822943411163&fdp=true): Provide Finance APIs with the ability to validate the writeable status of a deal in real time, with high availability

**Next actions (Rally):**

- Create dependency feature(s) in Deal Central to consume, once we’ve determined our approach