# MacOS MAC Address Randomization

<https://news.ycombinator.com/item?id=44338355>

sudo.exec("/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport en0 -z && ifconfig en0 ether \`openssl rand -hex 6 | sed 's/\\(..\\)/\\1:/g; s/.$//'\`",

<https://news.ycombinator.com/item?id=44310763>