#!/bin/bash
a=( 12 33 47 16 53 77 89 21 61 90 )
v=${#a[@]}
echo $v
for ((i=0; i<v; i++)); do
	for ((j=0; j<v-1; j++)); do
		if [ "${a[j+1]}" -gt "${a[j]}" ]; then
		temp=${a[j]}
		a[j]=${a[j+1]}
		a[j+1]=${temp}
		fi
		
done 

done
echo ${a[@]} 
	
