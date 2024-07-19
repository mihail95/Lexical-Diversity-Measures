import numpy as np
import string
from itertools import permutations

def generate_repeating_text_naive(vocabulary:set, length:int) -> str:
    """Generates a string out of a vocabulary of characters, without any randomness. Every character is used (approximately) the exact same amount of times (+- 1)"""
    generated = ""
    for ctr in range(length):
        generated += f"{list(vocabulary)[ctr % len(vocabulary)]} "

    return generated.strip()

def generate_random_text_naive(vocabulary:set, length:int) -> str:
    """Generates a string out of a vocabulary of characters using a completely random distribution."""
    vocab_list = list(vocabulary)
    rand_list = np.random.randint(0, len(vocab_list), length)
    generated_list = [ vocab_list[idx] for idx in rand_list ]
    generated = ' '.join(generated_list)

    return generated.strip()

def generate_random_text_zipf(vocabulary:set, length:int) -> str:
    """enerates a string out of a vocabulary of characters using a zipf distribution."""
    a = np.float64(1.2)
    minimum = np.uint64(1)
    maximum = np.uint64(len(vocabulary))
    zipf_list = zipf_distribution(a, minimum, maximum, length)

    generated_list = [ list(vocabulary)[idx-1] for idx in zipf_list ]
    generated = ' '.join(generated_list)
   
    return generated.strip()

def zipf_distribution(a: np.float64, min: np.uint64, max: np.uint64, size=None):
    """
    Generate Zipf-like random variables,
    but in inclusive [min...max] interval
    https://stackoverflow.com/a/57420941/6457269
    """
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(min, max+1, dtype=int) # values to sample
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)            # normalized

    return np.array(np.random.choice(v, size=size, replace=True, p=p))

def genearate_alphabet_permuatations(ngrams:int) -> set[str]:
    # For increasing text_length we need a bigger vocabulary
    alphabet = set(string.ascii_lowercase)
    perms = permutations(list(string.ascii_lowercase), ngrams)
    for perm in list(perms):
        alphabet.add(''.join(perm)) # Generates 672 unique tokens
    return alphabet