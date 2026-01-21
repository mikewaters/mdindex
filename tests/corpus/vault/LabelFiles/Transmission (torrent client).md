# Transmission (torrent client)

I needed some custom config to ensure the torrent downloads were bound to a VPN when port forwarding is not an option:

```shell

    local _t_addr=$(ifconfig ipsec0 inet | grep inet | awk '{print $2}')
    if [[ -z $_t_addr ]]; then
	echo "IPSEC VPN isnt enabled or bound"
	return 1
    fi
    echo "changing Transmission bind addr for IP4 to VPN: $_t_addr"
    defaults write org.m0k.transmission BindAddressIPv4 "$_t_addr"
    open -a Transmission
    _t_addr=$(ifconfig ipsec0 inet | grep inet | awk '{print $2}')
    echo "changing Transmission bind addr for IP4 back to local: $_t_addr"
    defaults write org.m0k.transmission BindAddressIPv4 "$_t_addr"
    echo "reset IP4 bind address. You should check open connections for IP6!"
    echo "This is all untested"
}
```

The key is `defaults write org.m0k.transmission BindAddressIPv4`, which i bind to the IPSEC VPN iface address (ip4). This may change though, and [Transmission *may* support this in the future](https://github.com/transmission/transmission/issues/338).