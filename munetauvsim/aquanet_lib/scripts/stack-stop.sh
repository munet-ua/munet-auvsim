#! /bin/sh
# Example of stopping all modules
# Modify the script as you needed
#                        by James
ps aux | grep "aquanet*" | grep `eval whoami` | grep -v grep | awk '{print $2}' | while read i; do kill $i; done
