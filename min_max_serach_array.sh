#!/bin/bash
a=( 10 23 45 56 67 78 89 90 15 18 07 99 14 )
echo "${a[@]}"
v="${#a[@]}"
echo "the length of the array is ", $v

min="${a[0]}"
max="${a[0]}"
for ((i=1; i<v; i++)); do
	if [ "${a[i]}" -le "$min"  ]; then
		min=${a[i]} 
	fi
	if [ "${a[i]}" -ge "$max" ]; then
	max="${a[i]}"
	fi
done
echo "the minimum value of array ",$min
echo "the maximum number in the array is ",$max
