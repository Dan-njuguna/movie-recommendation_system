#!/bin/bash

# AUTHOR: DAN NJUGUNA
# DATE: 13-04-2025

# Make required directories if they do not exist
DIRS=("data" "logs" "models" "db")
for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir $dir
    fi
done

# Install required packages
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found"
    exit 1
else
    pip install -r requirements.txt
fi


# Set up the database postgreSQL
# sudo -u postgres psql -c "CREATE DATABASE movie_recommendations;"
