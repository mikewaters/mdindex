# Using stored Keychain secrets in the shell

Rather than sticking OpenAI keys into `~.zshrc` (or `.zshenv`), use the keychain to pull the secrets using the `security` tool. Note: this is not the new Passwords app, you will only see it in Keychain as an “application password”.

> Don’t do this in zshrc, otherwise the vars will persist in your terminal session (fucking with any session timeouts you rely on).

### Adding an API key to keychain using the cli

#### 1\. add the secret

```shell
security add-generic-password -a "$USER" -s "OPENAI_API_KEY" -w "sk-your_key_here"
```

#### 2\. Allow your terminal app to access it 

This avoids two successive credential requests when using it.

TODO: cli-ify the instruction “Use the Keychain Access app”

### Retrieving it

```shell
security find-generic-password -a "$USER" -s "OPENAI_API_KEY" -w
```

### Use it in a script

```
export OPENAI_API_KEY=$(security find-generic-password -a "$USER" -s "OPENAI_API_KEY" -w)
```

### Making it on-demand

[Export secure credentials into the environment](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/c782355e-8878-42eb-bc52-4d46526d49e2#3a9fe8ca-a2e3-4c7d-9d0d-74a7b9a36ccc)

## Docs



```shell
>$ security add-generic-password
Usage: add-generic-password [-h] [-a account] [-s service] [-w password] [options...] [-A|-T appPath] [keychain]
    -a  Specify account name (required)
    -c  Specify item creator (optional four-character code)
    -C  Specify item type (optional four-character code)
    -D  Specify kind (default is "application password")
    -G  Specify generic attribute (optional)
    -j  Specify comment string (optional)
    -l  Specify label (if omitted, service name is used as default label)
    -s  Specify service name (required)
    -p  Specify password to be added (legacy option, equivalent to -w)
    -w  Specify password to be added
    -X  Specify password data to be added as a hexadecimal string
    -A  Allow any application to access this item without warning (insecure, not recommended!)
    -T  Specify an application which may access this item (multiple -T options are allowed)
    -U  Update item if it already exists (if omitted, the item cannot already exist)

By default, the application which creates an item is trusted to access its data without warning.
You can remove this default access by explicitly specifying an empty app pathname: -T ""
If no keychain is specified, the password is added to the default keychain.
Use of the -p or -w options is insecure. Specify -w as the last option to be prompted.

        Add a generic password item.
```