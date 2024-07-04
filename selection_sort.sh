#!/bin/bash
a=( 9 89 67 56 43 21 15 74 35 99 11 )
v=${#a[@]}
echo "the length "  $v
min=${a[0]}
#max=${a[0]}
for ((i=1; i<v; i++)); do
#echo $min
	for ((j=1; j<v; j++)); do 
		if [ "${a[j]}" -lt "$min" ]; then #gt for max 
		min=${a[j]}
		fi
		done
		done 
echo $min


#length of the array is 
#v=${#a[@]}
#echo "array lenght is ;", $v
#for ((i=0; i<v; i++)); do
	#for ((j=0; j<v-1; j++)); do
#		if [ "${a[j+1]}" -gt "${a[j]}" ] ; then
#			temp=${a[j]}
#			a[j+1]=${temp}
#			
#		fi
#	done
#done
#echo ${a[@]}
