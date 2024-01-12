import math

# this function is used to round the result to 2 decimal places
# e.g. 52.3523 -> 52.35, 52.0011 -> 52, 0.00000233 -> 0.0000023
def custom_round(x, decimal_places=2):
    str_x = f"{x:.10f}"
    before_decimal = str_x.split('.')[0]
    after_decimal = str_x.split('.')[1]
    leading_zeros = len(after_decimal) - len(after_decimal.lstrip('0'))
    
    if leading_zeros >= 1 and before_decimal == "0":
        return round(x, leading_zeros + 2)
    else:
        return round(x, decimal_places)

# this function converts a number in scientific notation to decimal notation
def scito_decimal(sci_str):
    def split_exponent(number_str):
        parts = number_str.split("e")
        coefficient = parts[0]
        exponent = int(parts[1]) if len(parts) == 2 else 0
        return coefficient, exponent

    def multiplyby_10(number_str, exponent):
        if exponent == 0:
            return number_str

        if exponent > 0:
            index = number_str.index(".") if "." in number_str else len(number_str)
            number_str = number_str.replace(".", "")
            new_index = index + exponent
            number_str += "0" * (new_index - len(number_str))
            if new_index < len(number_str):
                number_str = number_str[:new_index] + "." + number_str[new_index:]
            return number_str

        if exponent < 0:
            index = number_str.index(".") if "." in number_str else len(number_str)
            number_str = number_str.replace(".", "")
            new_index = index + exponent
            number_str = "0" * (-new_index) + number_str
            number_str = "0." + number_str
            return number_str

    coefficient, exponent = split_exponent(sci_str)
    decimal_str = multiplyby_10(coefficient, exponent)

    # remove trailing zeros
    if "." in decimal_str:
        decimal_str = decimal_str.rstrip("0")

    return decimal_str

# normalize the result to 2 decimal places and remove trailing zeros
def normalize(res, round_to=2):
        # we round the result to 2 decimal places
        res = custom_round(res, round_to)
        res = str(res)
        if "." in res:
            while res[-1] == "0":
                res = res[:-1]
            res = res.strip(".")
        
        # scientific notation
        if "e" in res:
            res = scito_decimal(res)

        return res

# 1. add
def add_(args):

    return normalize(sum(args))

# 2. subtract
def subtract_(args):

    res = args[0]
    for arg in args[1:]:
        res -= arg
    return normalize(res)

# 3. multiply
def multiply_(args):

    res = args[0]
    for arg in args[1:]:
        res *= arg
    return normalize(res)

# 4. divide
def divide_(args):

    res = args[0]
    for arg in args[1:]:
        res /= arg
    return normalize(res)

# 5. power
def power_(args):
        
    res = args[0]
    for arg in args[1:]:
        res **= arg
    return normalize(res)

# 6. square root
def sqrt_(args):
    res = args[0]
    return normalize(math.sqrt(res))

# 7. 10th log
def log_(args):
    # if only one argument is passed, it is 10th log
    if len(args) == 1:
        res = args[0]
        return normalize(math.log10(res))
    # if two arguments are passed, it is log with base as the second argument   
    elif len(args) == 2:
        res = args[0]
        base = args[1]
        return normalize(math.log(res, base))
    else:
        raise Exception("Invalid number of arguments passed to log function")

# 8. natural log
def ln_(args):
    res = args[0]
    return normalize(math.log(res))


# 9. choose
def choose_(args):
    n = args[0]
    r = args[1]
    return normalize(math.comb(n, r))

# 10. permutation
def permutate_(args):
    n = args[0]
    r = args[1]
    return normalize(math.perm(n, r))

# 11. greatest common divisor
def gcd_(args):
    res = args[0]
    for arg in args[1:]:
        res = math.gcd(res, arg)
    return normalize(res)

# 12. least common multiple
def lcm_(args):
    res = args[0]
    for arg in args[1:]:
        res = res * arg // math.gcd(res, arg)
    return normalize(res)

# 13. remainder
def remainder_(args):
    dividend = args[0]
    divisor = args[1]
    return normalize(dividend % divisor)