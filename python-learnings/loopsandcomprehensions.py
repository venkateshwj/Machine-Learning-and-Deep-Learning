#%%
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, sep='\n') # print all on same line

#%%
multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for multt in multiplicands:
    product = product * multt
product

#%%
s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for charr in s:
    if charr.isupper():
        print(charr, end='')

#%%
for i in range(5):
    print("Doing important work. i =", i)

#%%
r = range(5)
r

#%%
help(range)
list(range(5))

#%%
nums = [1, 2, 4, 8, 16]
for i in range(len(nums)):
    nums[i] = nums[i] * 2
nums

#%%
def double_odds(nums):
    for i, num in enumerate(nums):
        if num % 2 == 1:
            nums[i] = num * 2

x = list(range(10))
double_odds(x)
x   

#%%
list(enumerate(['a', 'b']))

x = 0.125
numerator, denominator = x.as_integer_ratio()

#%%
squares = [n**2 for n in range(10)]
squares

#%%
[32 for planet in planets]

#%%
def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    n_negative = 0
    for num in nums:
        if num < 0:
            n_negative = n_negative + 1
    return n_negative

