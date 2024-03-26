import random
import pandas as pd
from num2words import num2words
from scipy.stats import truncnorm


# Noisy data generation util functions

def generate_age():
    choice = random.choice(['integer', 'word'])
    if choice == 'integer':
        return random.randint(1, 100)  # Generate random integer age
    elif choice == 'word':
        return num2words(random.randint(1, 100))


def generate_height_weight():
    a = (1.3 - 1.7) / 0.2
    b = (2.3 - 1.7) / 0.2
    choice = random.choice([';', ',', '-', '_'])
    words = random.choice([True, False])
    height = truncnorm.rvs(a, b, loc=1.7, scale=0.2, size=1)[0]
    weight = random.randrange(40, 150)
    if words:
        weight = num2words(weight)
    return str(height) + choice + str(weight)


def generate_age_column(n=100):
    return pd.Series([generate_age() for _ in range(n)])


def generate_height_weight_column(n=100):
    return pd.Series([generate_height_weight() for _ in range(n)])


def generate_categorical_column(categories, n=100):
    return pd.Series([random.choice(categories) for _ in range(n)])


def generate_missing_numeric_column(minv=1000, maxv=10000, n=100):
    return pd.Series([random.choice([random.randint(minv, maxv), None]) for _ in range(n)])


def generate_yes_no_column(n=100):
    return pd.Series([random.choice(['yes', 'no']) for _ in range(n)])

