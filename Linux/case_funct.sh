#!/bin/bash

delete() {
    read -p "Enter the file to delete: " file_to_delete
    if [ -e "$file_to_delete" ]; then
        rm -i "$file_to_delete"
        echo "$file_to_delete file is deleted"
    else
        echo "$file_to_delete is not found"
    fi
}

copy_the_file() {
    read -p "Enter the filename to copy: " name
    if [ -e "$name" ]; then
        cp -r "$name" "$name.bak"
        echo "The file is copied to $name.bak"
    else
        echo "$name file is not found"
    fi
}

rename_file() {
    read -p "Enter the name of the file to rename: " old_name
    read -p "Enter the new name for the file: " new_name
    if [ -e "$old_name" ]; then
        mv "$old_name" "$new_name"
        echo "The file is renamed to $new_name"
    else
        echo "$old_name file is not found"
    fi
}

create_file() {
    read -p "Enter the name of the file to create: " filename_create
    touch "$filename_create"
    echo "$filename_create file is created"
}

list_files() {
    read -p "Enter the directory name for listing: " dir_name
    if [ -d "$dir_name" ]; then
        ls -l "$dir_name"
        echo "The list of files in the directory"
    else
        echo "$dir_name is not a valid directory or does not exist"
    fi
}

help_message() {
    echo "Usage: $0 <choice>"
    echo "Choices:"
    echo "  1. Delete"
    echo "  2. Copy"
    echo "  3. Rename"
    echo "  4. Create"
    echo "  5. List"
}

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        delete
        ;;
    2)
        copy_the_file
        ;;
    3)
        rename_file
        ;;
    4)
        create_file
        ;;
    5)
        list_files
        ;;
    *)
        echo "Invalid choice. Please enter a number between 1 and 5."
        help_message
        ;;
esac

