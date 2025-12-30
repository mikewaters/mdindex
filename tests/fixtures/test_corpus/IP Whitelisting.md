# IP Whitelisting

## For Credit Bureau

???

## Globally for uniFI

Page loads as well as menu items can be IP-excluded.

Implemented in [Common Objects](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/3d8a09a3-96f1-4769-97cb-3ea61b82b958)

Used in [menu access check](https://ghe.coxautoinc.com/DT-SFI/fni-commonobjects/blob/d53c7662db3b91114d82f4fafc5f56af9752364b/CO_CMN/COMPONENTSNET/DealerTrack.CommonObjects.Menu/clsMenu.vb#L1979) (Classic)

Used in [dealer switch](https://ghe.coxautoinc.com/DT-SFI/fni-commonobjectsV2/blob/d7a978072dc1137101511218c14a23587acf0fb9/CO_CMN/COMPONENTSNET/DealerTrack.CommonObjects.DealerProfile/clsDealerProfile.vb#L466)



## Route Permissions

uniFI can apply an [IP whitelist protection feature](https://ghe.coxautoinc.com/DT-SFI/dt/blob/92dee8c44102907167360c76d96a1d374ed53f33/dtplatform/dtplatform/core/security_manager.py#L194-L206) to an individual route, using the \``ip_restriction_type_code`for that router.

[SecurityManager](https://ghe.coxautoinc.com/DT-SFI/dt/blob/92dee8c44102907167360c76d96a1d374ed53f33/dtplatform/dtplatform/core/security_manager.py#L158) code

## CLR Routing

For Oauth, on CLR [PageLoad](https://ghe.coxautoinc.com/DT-SFI/Admin-ClrRouting/blob/6e9bd09f3815a4792e958f451996d72c7d75db63/ClrRouting/routing.aspx.cs#L51)

[CLR log message production](https://ghe.coxautoinc.com/DT-SFI/Admin-Clr.Routing/blob/79843ab5566685d4797cfbb452371a844682cba2/ClrRouting/routing.aspx.cs)



# Splunk queries

[Parsing an existing field using a regex and pulling out a new field](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/634ece52-8c33-4c20-ad6a-964a1fc955b6#9df43a70-3cd0-4dad-8135-7d61fb9a3d7e)

### Find CLR ip restriction logs

> `dt-sfi-indexes` "iprestriction=Y"

### Find SecurityManager restriction logs, by route

> index=dt-sfi_app "IP Restriction," |spath "line.message.extra.route_name" output=route 

### Find IP restriction failures globally

> `dt-sfi-indexes` iprestrictmsg NOT "iprestrictmsg=USER is not enabled" NOT "is in within the range" NOT "iprestrictmsg=There are no IP"