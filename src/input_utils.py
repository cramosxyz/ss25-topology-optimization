def get_valid_input(prompt, type_func, error_message):
    """
    Prompts the user for numeric input, validating that it's a valid type.

    Args:
        prompt (str): The prompt to display to the user.
        type_func (function): A function that attempts to convert the input to the desired type (e.g., int(), float()).
        error_message (str): The error message to display if the conversion fails.

    Returns:
        int or float: The validated numeric input, or None if the input is invalid.
    """
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            print("Input cannot be empty. Please try again.")
            continue

        try:
            value = type_func(user_input)
            return value
        except ValueError:
            print(error_message)
            continue