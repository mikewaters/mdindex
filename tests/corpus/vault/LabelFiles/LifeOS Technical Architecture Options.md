# LifeOS Technical Architecture Options

## Hosting Options

### Application services

1. Local (Mac Mini, NAS, or some future mini pc tbd

2. [Infrastructure Hosting Options.md](./Infrastructure%20Hosting%20Options.md)

### Databases

1. Locally, even running local supabase stack

2. [Database Hosting Options.md](./Database%20Hosting%20Options.md)

## Architecture Options

1. Remote database + mobile app

   1. Mobile app must be online

2. Remote database + mobile app with local replica using online writes

   1. Mobile app must be online for writes

   2. Mobile app must be online to receive updates

3. Remote database + mobile app with local replica using local writes

   1. Mobile app must be online to receive updates

## Database Options

[Backend Data Frameworks.md](./Backend%20Data%20Frameworks.md)

[Middleware Sync Frameworks.md](./Middleware%20Sync%20Frameworks.md)

1. Postgres + ElectricSQL

   - Managed postgres with ElectricSQL hosted

      - Supabase or other

      - Digital Ocean

   - Unmanaged Postgres with ElectricSQL hosted

      - Supabase on DO, install electric on same VM

      - Raw PG, with electric on the same VM

2. Trailbase (no ElectricSQL)

3. Turso

## Mobile App Options

[Creating an iOS App.md](./Creating%20an%20iOS%20App.md)

[Frontend Sync Frameworks.md](./Frontend%20Sync%20Frameworks.md)

1. LibSQL client

2. TanStack Query/DB (Electric, Trailbase, any REST API like Supabase)