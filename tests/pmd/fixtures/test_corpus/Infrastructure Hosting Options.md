# Infrastructure Hosting Options

### Digital Ocean

- [Officially supported](https://electric-sql.com/docs/integrations/digital-ocean) by ElectricSQL

- Because I am using essentially a VM, I can install Tailscale, install my own Postgres, etc. They even have a managed postgres, but I am not sure what the costs are compared to neon or supabase etc

- [DigitalOcean - Deploying a static Astro web site with Tailscale, Docker, and Caddy by Andrew Hoog](https://www.andrewhoog.com/posts/deploying-a-static-astro-web-site-on-digitalocean-with-tailscale-docker-and-caddy/)

- Managed Postgres is $15/month, Droplets are $4/month

- Has a Supabase droplet that comes with an embedded postgres. Has all the edge functions and realtime embedded, might be a cheap option to kick off

### [Fly.io](Fly.io)

- [Supports Tailscale natively](https://tailscale.com/kb/1132/flydotio)

- Can run app server (like Electric) as well as a Postgres database

- Has a managed Postgres option, but it does not support community plugins beyond pgvector and one other

   - [Fly Docs - Managed Postgres](https://fly.io/docs/mpg/)

   - $15/month, no free tier