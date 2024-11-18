#!/bin/bash
find . -maxdepth 1 -type f -regextype posix-extended -regex '.*\.(jpg|jpeg|png|gif|webp)$' -printf '%f\n' | sort -V | python -c "import sys; print(f'\"{[line.strip() for line in sys.stdin]}\"')"
# EOF
