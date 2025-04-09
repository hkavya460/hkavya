#!/bin/bash
array=(2 6 8 9 45 67 3 25)
#echo ${array[*]}
#echo ${array[0]}
v=${#array[@]}
echo $v

for ((i=0; i<v; i++)); do
    for ((j=0; j<v-1; j++)); do
        if [ ${array[j+1]} -lt ${array[j]} ]; then
            # Swap elements
            temp=${array[j]}
            array[j]=${array[j+1]}
            array[j+1]=$temp
        fi
    done
done

# Print sorted array
echo "${array[@]}"

