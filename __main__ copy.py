from math import log10


def hybrid_ints():
    running = True
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    p, q = primes[0], primes[0]
    i = 0
    count = 0
    while running:
        if i < len(primes) - 1:
            p = primes[i]
        else:
            not_found_prime = True
            while not_found_prime:
                not_prime = False
                p += 2
                for prime in primes:
                    if p % prime == 0:
                        not_prime = True
                        break
                if not_prime:
                    continue
                primes.append(p)
                # print(primes)
                break
        print(p, count)
        while True:
            if q * log10(p) + p * log10(q) < 47229443:
                count += 1
            else:
                break
        # if i > 100:
        #     running = False
        #     break
        i += 1
    return count


print(hybrid_ints())
