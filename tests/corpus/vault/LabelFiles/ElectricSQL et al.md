# ElectricSQL et al

<https://news.ycombinator.com/item?id=39977090>

> The paradigm shift that's going to come with distributed SQLite isn't to the edge, it's going to be to users devices. I believe the DX of building Local-first apps is going to hit the criticality point in the next year or so and its popularity is going to explode.
>
>  With dynamic partial replication you can synchronise a subset of your database to a SQLite db on your users device, eliminating the network from the ui interaction loop. Then with the emerging eventually constant syncing systems, many using CRDTs, it's possible to have conflict free eventual consistency. Just read and write to a local database and it will sync with your central server or other clients in the background, both in realtime or after working offline.
>
> I work on one such system at ElectricSQL, but there are many people building variants of this such as Evolu, SQLsync, CR-SQLite, and PowerSync.
>
> That's all not to say SQLite on the edge isn't really damn cool!