#captalize the letters
fruits =  ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
capitalized_fruits = []
for fruits in fruits:
    capitalized = (fruits.capitalize())
    capitalized_fruits.append(capitalized)
print( capitalized_fruits )
#capitalized= [fruits.capitalize() for fruit in fruits]
print(capitalized)

2#Make a variable named fruits_with_only_two_vowels
fruits_with_only_two_vowels =[ fruit for fruit  in fruits if (
    fruit.count("a") + 
    fruit.count("e") + 
    fruit.count("i") + 
    fruit.count("o") + 
    fruit.count("u")) == 2]
print(fruits_with_only_two_vowels)

import  difflib
#3]sequence similarity
#pair all possible sequences of org1 and org2
#compare both and for similar base within the seq give 1 and dissimilar give 0 add the given values:
#similarity = sum_of_values / len of seq * 100
# set threshold to 80%
#if similarity < 80% print sequences
#else consider next pair

# Calculate the similarity between seq1 and seq2 using the SequenceMatcher
def similarity(seq1, seq2, threshold):
    seq_matcher = difflib.SequenceMatcher(None,seq1, seq2)
    similarity = seq_matcher.ratio()
    return similarity > threshold

org1 = ["ACGTTTCA", "AGGCCTTA", "AAAACCTG"]
org2 = ["AGCTTTGA", "GCCGGAAT", "GCTACTGA"]
threshold = 0.6  # Define your desired similarity threshold

similar_pairs = [(seq1, seq2) for seq1 in org1 for seq2 in org2 if similarity(seq1, seq2, threshold)]
print(similar_pairs)


#dictionary of numbers and their squares, excluding odd numbers using dictionary comprehension

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

odd_square_dict = {number:number * number for number  in numbers if number % 2 != 0}
print(odd_square_dict)

#dictionary comprehension to map words to their reverse in a sentence.
sentence = "Hello, how are you?"
new_sentence = sentence.split(' ')
reversed_str_dict = {X:X[::-1] for X in new_sentence}
print(reversed_str_dict)

#Write  a lambda function to sort a list of strings by the last character

list = (["bindu", "kavya", "farisaf"])
new_list = sorted(list, key=lambda x: x[-1])
new_list = list.sort(key = lambda x : x[-1])
print(new_list)

# rearrange positive and negative numnbers in a give array using lambda
arr = [1,0,4,6,7,-5,32,-9]
rearranged_arr = sorted(arr, key = lambda x : -x)
print(rearranged_arr)

#add(2,3), create a decorator that prints the following: Calling add with args: (2, 3), kwargs: {}
def log_function(func):
    def wrap(*args,**kwargs):
        print(f"calling{func.__name__} with args:{args},kwargs:{kwargs}")
        result = func(*args,**kwargs)
        print(f"{func.__name__} returned:{result}")
        return result
    return wrap
@log_function
def addition(a,b):
    return a + b
result=addition(2,3)
print("result :", result)


#9]Create a decorator to measure the execution time of a function
import time
import math
def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:} seconds to execute.")
        return result
    return wrapper
@measure_execution_time
def fact(num):
    time.sleep(1)
    print(math.factorial(num))

fact(5)


#10] try and except
def division(a, b):
    try:
        result = a / b
        print(f"{a} / {b} = {result}")
    except ZeroDivisionError:
        print('denominator should not be zero')
    except ValueError:
        print("invalid input,can't devide")
    except Exception as e:
       print('unknown error!', e)
    finally:
        print("done")

def main():
    a = 46
    b = 2
    division(a, b)

    a = 46
    b = 0
    division(a, b)

    a = "a"
    b = 2
    division(a, b)


if __name__ == "__main__":
    main()

#11]calculator and formula error
class FormulaError(Exception):
    pass

def inputs(user_input):
  input_list = user_input.split()
  if len(input_list) != 3:
    raise FormulaError('Input does not consist of three elements')
  a, op, b = input_list
  try:
    a = float(a)
    b = float(b)
  except ValueError:
    raise FormulaError('The first and third input value must be numbers')
  return a, op, b


def calculate(a, op, b):
  if op == '+':
    return a + b
  if op == '-':
    return a - b
  raise FormulaError('{0} is not a valid operator'.format(op))

try:
    user_input = input("Enter a mathematical expression: ")
    a, op, b = inputs(user_input)
    result = calculate(a, op, b)
    print(f"Result: {result}")
except FormulaError as e:
    print(f"Error: {e}")












