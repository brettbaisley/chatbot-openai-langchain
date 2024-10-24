# Fixlog for Alert

## Alert Summary
* Host abcxyz high memory usage

## Background
* Some processes that run on the host are using a lot of memory.

## Resolution Steps
1. Run `top` to see the processes running.
2. Run `free -m` to check the memory usage.
3. If the memory usage is high, check the processes using the most memory.
4. Kill the processes using the most memory.
5. If the memory usage is still high, reboot the host.
6. If the alert comes back, contact the support team.
