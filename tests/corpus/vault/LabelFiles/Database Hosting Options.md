# Database Hosting Options

### Turso

SQlite only

Supports read replicas, but requires a js or rust client sdk for local client writes (requires online writes for everything else - like most other frameworks)

### Supabase 

Supported by ElectricSQL

Free tier, then $25/month

#### Open source hosted on Digital Ocean

Can either use the DO one-click droplet, which installs everything to the VM, I could use a docker droplet and run Supabase using its docker-compose-based self-hosting instructions.

Whichever way I do it, I will need to put the postgres on a dedicated volume - if I dont use a managed postgres for this

A) DO one-click droplet, which installs Supabase to the VM:

[DigitalOcean Marketplace - Supabase 1-Click App](https://marketplace.digitalocean.com/apps/supabase)

Installer Packer code:

<https://github.com/digitalocean/droplet-1-clicks/tree/master/supabase-22-04>

B) Supabase install instructions using docker-compose:

<https://supabase.com/docs/guides/self-hosting/docker>

DO docker droplet:

[DigitalOcean Marketplace - Supabase 1-Click App](https://marketplace.digitalocean.com/apps/docker)

### Fly Postgres

[Fly.io](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/35cde992-64ef-479e-a8d1-e7a234b24543#87f31da9-ef6b-46bd-b423-95016c001bc7) $15/month, no free tier

### Digital Ocean Postgres

[DO](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/35cde992-64ef-479e-a8d1-e7a234b24543#bd7e52d2-8044-47e4-9fea-5efef783c8b4) $25/month, no free tier

### Neon Serverless Postgres

Free tier, then usage-based



# 