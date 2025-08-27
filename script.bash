bash
#!/bin/bash

# Loop over gz archives
find . -type f -name "*.gz" | while read -r archive; do
    # Create a temporary directory
    tmpdir=$(mktemp -d)

    # Try to extract
    tar -xzf "$archive" -C "$tmpdir" 2>/dev/null || gzip -cd "$archive" > "$tmpdir/file" 2>/dev/null

    # Find python files inside the extracted content
    find "$tmpdir" -type f -name "*.py" | while read -r pyfile; do
        clear
        echo "Opening: $pyfile from $archive"
        nano "$pyfile" # let user inspect
        read -p "Is this the file? Yes/no: " answer
        case "$answer" in
            [Yy]* )
                echo "Confirmed. Keeping $archive, removing all others..."
                find . -type f -name "*.gz" ! -path "$archive" -delete
                rm -rf "$tmpdir"
                exit 0
                ;;
            * )
                echo "Skipping..."
                ;;
        esac
    done
    rm -rf "$tmpdir"
done

echo "No matching file confirmed."
