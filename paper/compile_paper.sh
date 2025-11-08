#!/bin/bash

# Script to compile pymars paper for submission

# Check if quarto is installed
if ! command -v quarto &> /dev/null
then
    echo "Quarto is not installed. Please install quarto to compile the paper."
    echo "Visit https://quarto.org/docs/get-started/ for installation instructions."
    exit 1
fi

# Compile the revised paper to PDF
echo "Compiling pymars paper to PDF..."
quarto render pymars-paper-revised.qmd --to pdf

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Paper compiled successfully!"
    echo "Output file: pymars-paper-revised.pdf"
else
    echo "Paper compilation failed!"
    exit 1
fi

# Also compile to HTML for GitHub Pages
echo "Compiling paper to HTML for GitHub Pages..."
quarto render pymars-paper-revised.qmd --to html

if [ $? -eq 0 ]; then
    echo "HTML version compiled successfully!"
    echo "Output file: pymars-paper-revised.html"
else
    echo "HTML compilation failed!"
fi