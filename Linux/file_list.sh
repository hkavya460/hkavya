#!/bin/bash

delete() {
read -p "enter the file to delete:" file_to_delete
if [ -e "$file_to_delete" ]; then
rm -i "$file_to_delete" 
echo "$file_to_delete file is deleted"
else
echo "$file_to_delete is not found"
fi
}
copy_the_file() {

read -p "enter the filename to copy:" name
if [ -e "$name" ]; then
cp -r "$name""$name.bak"
echo "the file is copied to $name.bak"
else
echo "$name file is not found"
fi
}
rename_file() {
read -p "enter the name for the file to rename:" old_file
read -p "enter new name for the file :" new_file 
if [ -e "$old_file" ];then
mv "$old_file" "$new_file"
echo "the file is renamed to  $new_file "
else
echo "$old_file file is not exist"
fi
}
create() {   #check for the file is present or not {
read -p  "enter the name of file to create:"filename_create
touch "$filename_create"
echo " $filename_create file is created"
}
list_files()
{
read -p "enter the directory name for listing:" dir_name
if [ -d "$dir_name" ]; then
ls -l "$dir_name"
echo "the list of the directories"
else
echo "$dir_name is not valid directorary or directorary does not exist"

fi
}

help_message() {
echo "usage:$0 <choice>"
echo "Choices:"
    echo "  1. Delete"
    echo "  2. Copy"
    echo "  3. Rename"
    echo "  4. Create"
    echo "  5. List"
 }
read -p "enter the choice (1-5):" choice
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
echo "invalid choice please enter the value in between 1 and 5 "
	help_message
;;
esac


