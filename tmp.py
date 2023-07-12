from concurrent.futures import ThreadPoolExecutor, as_completed


def test(hello, world):
    return hello + world

def main(a, b):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = []
        results2 = []

        future_to_stuff = [executor.submit(test, hello, world) 
                           for hello, world in zip(a, b)]
        for future in future_to_stuff:
            results2.append(future.result())

    print(results)
    print(results2)

if __name__ == '__main__':
    a = [1, 2, 3, 4]
    b = [1, 2, 3, 4]
    main(a, b)