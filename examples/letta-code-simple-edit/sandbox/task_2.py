def find_last_element(arr):
    """Return the last element of an array."""
    if len(arr) == 0:
        return None
    return arr[len(arr) - 1]


def main():
    my_list = [10, 20, 30, 40, 50]
    last = find_last_element(my_list)
    print(f"The last element is: {last}")


if __name__ == "__main__":
    main()
