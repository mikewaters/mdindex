---
tags:
  - landscape
Document Type:
  - Technical
Topic:
  - software
Subject:
  - Solution
---
# Management Dashboards Landscape

Excluding commercial solutions like NewRelic or large scale solutions like Prometheus/Grafana.

## Solution Arcs

There are a few wide arcs of dashboard type, each solution sits in a venn diagram somewhere.

### “Start pages”

Designed to be your browser homepage, and vary from static sites that show browser favorites to full-on hosted systems.

### Service Dashboards

A step up from bookmarks, this is where you will have widgets for all the web services or apps you use, whether they are self-hosted or not. Solutions typically have canned integrations for a wide array of services, like Pi-Hole, Gitlab etc

### Monitoring Dashboards

Focused more on system monitoring and availability, the NewRelics of the world. Either push or pull-based, relying the availability of specific metrics to aggregate centrally.

### Status Pages

Focused more on polling of arbitrary properties, to measure uptime or service availability. Doesnt require access to system resource metrics.

## Start Pages

### Glance

> self-hosted dashboard that puts all your feeds in one place

Golang 

<https://github.com/glanceapp/glance>

Has a mobile experience

Static; Does not have any data refresh or sync, only updates on page reload

Eval: 

- More for RSS and social media feeds, with a bit of integrations added on.

- I really like the look, its terminal-esque

## Service Dashboards

Service- or application-level monitoring and integrations. Not focused on metrics per-se, but can display them. Has other stuff like feeds, bookmarks etc.

### Homepage

[gethomepage / homepage](https://github.com/gethomepage/homepage)

> A highly customizable homepage (or startpage / application dashboard) with Docker and service API integrations.

Very much tied to Docker, the personal stuff seems like an addon. So this is more of a system monitor++

Calls out being **static**, and so expectation is refresh is needed for updates.

### Dashy

> A self-hostable personal dashboard built for you. Includes status-checking, widgets, themes, icon packs, a UI editor and tons more!

> Dashy is an open source, highly customizable, easy to use, privacy-respecting dashboard app.
>
> [dashy.to](dashy.to)

Demo: <https://demo.dashy.to/> 

Showcase: <https://github.com/Lissy93/dashy/blob/master/docs/showcase.md>

Widgets: <https://github.com/Lissy93/dashy/blob/master/docs/widgets.md>

Eval:

- Fuckin ugly

### Homarr

Eval: Cool, more in the “\*arr” world, and meant for babies.

### ~~Flame~~

~~<https://github.com/pawelmalak/flame>~~

Eval: Not updated in some time

### A bunch of PHP crap

- Heimdall

- Organizr

> Not even bothering

## Metrics Collection Dashboards

Agent- or client-based collection of system metrics for aggregation.

### Bezsel

> Beszel is a lightweight server monitoring platform that includes Docker statistics, historical data, and alert functions.
>
> It has a friendly web interface, simple configuration, and is ready to use out of the box. It supports automatic backup, multi-user, OAuth authentication, and API access.

Good as a compact, single-pane server monitor for Docker hosts where you want metrics + alerts without the Prometheus+Grafana stack

<https://github.com/henrygd/beszel>

Eval: Loves it, but its not good enough on its own. Will use it for metrics collection.

### ~~Xpipe~~

~~<https://github.com/xpipe-io/xpipe>~~

**Java, desktop app**

> ~~XPipe is a new type of shell connection hub and remote file manager that allows you to access your entire server infrastructure from your local machine. It works on top of your installed command-line programs and does not require any setup on your remote systems. So if you normally use CLI tools like `ssh`, `docker`, `kubectl`, etc. to connect to your servers, you can just use XPipe on top of that.~~

Eval: Yuck.

## Lab Metrics Agents

Monitoring metrics like cpu, network etc. 

### Beszel

[henrygd / beszel](https://github.com/henrygd/beszel)

Lightweight server monitoring hub with historical data, docker stats, and alerts.

### Glances

<https://github.com/nicolargo/glances>

Has a web server

Python/fastapi

Uptime monitoring

*not* “Glance”

Eval: nahh

## Status Pages

### Uptime Kuma

> Uptime Kuma is an easy-to-use self-hosted monitoring tool.
>
> Monitoring uptime for HTTP(s) / TCP / HTTP(s) Keyword / HTTP(s) Json Query / Ping / DNS Record / Push / Steam Game Server / Docker Containers

<https://github.com/louislam/uptime-kuma>

Eval: Cool AF

## Resources

- <https://github.com/ZhaoUncle/Awesome-Homepage>

- <https://github.com/jnmcfly/awesome-startpage>