def calculate_sum(numbers)
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total


if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5]
    result = calculate_sum(nums)
    print(f"The sum is: {result}")
