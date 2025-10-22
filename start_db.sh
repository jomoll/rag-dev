#!/bin/bash

# Use the working temp directories under /data/moll
export TMPDIR=/data/moll/datasette_temp
export TEMP=/data/moll/datasette_temp
export TMP=/data/moll/datasette_temp
export XDG_CACHE_HOME=/data/moll/datasette_cache

# Create directories
mkdir -p $TMPDIR
mkdir -p $XDG_CACHE_HOME

# Clean any existing temp files
rm -rf $TMPDIR/*

echo "Using temp directory: $TMPDIR"
echo "Using cache directory: $XDG_CACHE_HOME"

# Check available space
df -h $TMPDIR

# Start Datasette with the database
cd /home/moll/rag
datasette myeloma_reports_de.sqlite \
    -p 8521 \
    --setting sql_time_limit_ms 30000 \
    --setting max_returned_rows 1000 \
    --setting allow_download on \
    --setting cache_size_kb 2000 \
    --cors