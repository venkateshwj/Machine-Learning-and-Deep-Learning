#%%
primes = [2, 3, 5, 7]

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'], # (Comma after the last element is optional)
]
# (I could also have written this on one line, but it can get hard to read)
hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]

my_favourite_things = [32, 'raindrops on roses', help]
my_favourite_things[1:]
#%%
my_favourite_things[-2:]
#%%
my_favourite_things[1]='hi'
#%%
len(my_favourite_things)
#%%

sorted(planets)

#%%
x = 13
# x is a real number, so its imaginary part is 0.
print(x.imag)
# Here's how to make a complex number, in case you've ever been curious:
c = 12 + 3j
print(c.imag)
#%%
yyy=100
yyy.bit_length()

#%%
my_favourite_things.append('jii')
my_favourite_things
#%%
x = 0.100
x.as_integer_ratio()

numerator, denominator = x.as_integer_ratio()
print(numerator / denominator)




