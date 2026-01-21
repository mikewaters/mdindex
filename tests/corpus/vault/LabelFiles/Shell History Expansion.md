---
tags:
  - document 📑
---
# Shell History Expansion

"bangs"

1. Common History Operators:

```bash
!!      # Entire last command
!-2     # Two commands ago
!$      # Last argument of previous command
!^      # First argument of previous command
!*      # All arguments of previous command
!:2     # Second argument of previous command
!:2-4   # Arguments 2-4 of previous command
!:2*    # From second argument to last
!:2-$   # Same as above (from 2nd to last)
!?string? # Last command containing "string"
```

1. Word Modifiers (add : and modifier after any of above):

```bash
!!:h    # Remove trailing pathname component
!!:t    # Remove all leading pathname components
!!:r    # Remove trailing suffix
!!:e    # Remove all but trailing suffix
!!:p    # Print command without executing
!!:s/old/new/  # Substitute first occurrence
!!:gs/old/new/ # Substitute globally
```

1. Quick Substitutions:

```bash
^old^new   # Replace 'old' with 'new' in last command
!!:s/old/new/  # Same as above
```

1. Directory Navigation:

```bash
cd -      # Previous directory
cd ~-2    # Directory from 2 positions ago in stack
dirs -v   # List directory stack
pushd +n  # Rotate stack n times
```

1. Practical Examples:

```bash
# Rerun last command with sudo
sudo !!

# Move file you just created
touch myfile.txt
mv !$ myfile2.txt

# Use argument from specific position in history
ls /very/long/path/
cd !-1:$  # Uses last argument from previous command

# Quick find and replace in last command
ls -l /path/to/wrong/file
^wrong^right  # Runs: ls -l /path/to/right/file

# Print matched command before executing
!ssh:p    # Print last command starting with 'ssh'
```

1. Special Parameters:

```bash
$_    # Last argument of previous command
$0    # Name of shell or script
$#    # Number of parameters
$@    # All parameters (preserves quoting)
$*    # All parameters (single string)
$$    # Current shell/process ID
$?    # Exit status of last command
$!    # PID of last background job
```

1. Quick Reference Shortcuts:

```bash
ESC .   # Insert last argument (can press multiple times)
Alt+.   # Same as above (some terminals)
Ctrl+R  # Reverse history search
!#      # Current command line up to cursor
```

Tips:

- Use `history` to see command history

- Add `setopt HIST_VERIFY` to your .zshrc to see expansions before executing