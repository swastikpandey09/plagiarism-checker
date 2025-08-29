#!/bin/bash
find . -type f -name "*.gz" | while read -r archive; do
    tmpdir=$(mktemp -d)
    tar -xzf "$archive" -C "$tmpdir" 2>/dev/null || gzip -cd "$archive" > "$tmpdir/file" 2>/dev/null
    find "$tmpdir" -type f -name "*.py" | while read -r pyfile; do
        clear
        echo "===== Showing: $pyfile from $archive ====="
        echo
        cat "$pyfile"
        echo
        echo "=========================================="
        read -p "Press Enter when you are done reading..." dummy
        read -p "this is the file Yes/no: " answer
        case "$answer" in
            [Yy]*) echo "Confirmed. Keeping $archive, removing all others..."
                   find . -type f -name "*.gz" ! -path "$archive" -delete
                   rm -rf "$tmpdir"
                   exit 0 ;;
            *) echo "Skipping..." ;;
        esac
    done
    rm -rf "$tmpdir"
done
echo "No matching file confirmed."
