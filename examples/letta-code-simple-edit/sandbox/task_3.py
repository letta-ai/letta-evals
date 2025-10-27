def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return None
    total = sum(numbers)
    count = len(numbers)
    return total / count


def main():
    # test with empty list
    empty_list = []
    avg = calculate_average(empty_list)
    print(f"Average: {avg}")

    # test with normal list
    nums = [10, 20, 30, 40, 50]
    avg = calculate_average(nums)
    print(f"Average: {avg}")


if __name__ == "__main__":
    main()
