#!/bin/bash
#var1=10
#var2=20 
#sum=$(($var1  +  $var2))
#echo $sum
#echo "the difference is",  $diff
#bubble sort :a rray of elemts in descending order
array=( 2 6 8 9 45 67 3 25 55 77 108 )
echo ${array[*]}
echo ${array[0]}
v=${#array[@]}
echo $v
for ((i=0; i<v; i++)); do
	for ((j=0; j<v-1; j++)); do
		 if [ ${array[j+1]} -gt ${array[j]} ]; then

			temp=${array[j]}
			array[j]=${array[j+1]}
			array[j+1]=${temp}
		fi
	done 

done 
echo "${array[@]}"
#echo "${array[3]}"
#insertion sort
#fibanocci number
#a=0
#b=1
#echo $a
#echo $b
#n=15
#i=2
#while [ $i -le 15 ];do

#	c=$(($a+$b))
#	echo $c
#	a=$b
#	b=$c
#	i=$((i+1))
#done
#onther type
#a=0
#b=1
#echo $a
#echo $b
#n=10
#for ((i=2; i<=$n+1; i++)); do
# c=$(($a +$b))
# echo $c
# a=$b
# b=$c
#done
#factorial 
 
#23 factorial 23!=23*22*21!
#n=8
#echo $n
#fact=1
#echo $fact
#for ((i =1; i<=$n; i++)); do
	#fact=$((fact * i))
	#n--
#done
#echo "$fact"
#number =24
#echo $number
#for (( i=2; i-ge $number; i++ )); do
#	$i % 2 ==1

#done 
#echo $number is prime
#!/bin/bash
a=(10 23 45 56 67 78 89 90 15 18 7 99 14)
echo "Original array: ${a[@]}"
v="${#a[@]}"
echo "Number of elements in the array: $v"

# Initialize min with the first element of the array
min="${a[0]}"

# Iterate through the array to find the minimum value
for ((i=1; i<v; i++)); do
    if [ ${a[i]} -le $min ]; then
        min=${a[i]}
    fi
done

echo "Minimum value in the array: $min"

	



