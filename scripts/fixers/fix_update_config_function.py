#!/usr/bin/env python3
"""
Fix the _update_config_value function in web_scraper.py
"""

def fix_update_config_function():
    with open('wdbx_plugins/web_scraper.py', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Define the problematic function that needs to be fixed
    old_function = r"""def _update_config_value\(key: str, value: str\) -> bool:.*?def _normalize_url"""
    
    # Create the fixed version of the function
    new_function = """def _update_config_value(key: str, value: str) -> bool:
    \"\"\"
    Update a single config value with type conversion and validation.

    Args:
        key: Config key to update
        value: New value (as string)

    Returns:
        True if update was successful, False otherwise
    \"\"\"
    if not hasattr(scraper_config, key):
        print(f"\\033[1;31mError: Unknown configuration key: {key}\\033[0m")
        return False

    # Get the expected type from the current config attribute
    expected_type = type(getattr(scraper_config, key))
    original_value = getattr(scraper_config, key)
    new_value = None

    try:
        if expected_type == bool:
            new_value = value.lower() in ("true", "yes", "1", "y")
        elif expected_type == int:
            new_value = int(value)
        elif expected_type == str:
            new_value = value
        elif expected_type == list:
            if value.startswith("[") and value.endswith("]"):
                new_value = json.loads(value)
            else:
                new_value = [x.strip() for x in value.split(",")]
            if not isinstance(new_value, list):
                raise ValueError("Invalid list format")
        else:
            error_msg = f"Cannot update configuration key '{key}' of type {expected_type.__name__}"
            print(f"\\033[1;31m{error_msg}\\033[0m")
            return False

        # Temporarily update the attribute
        setattr(scraper_config, key, new_value)

        # Re-validate the entire config object
        try:
            scraper_config.__post_init__()  # Trigger validation
            print(f"\\033[1;32mSet {key} = {new_value}\\033[0m")
            return True
        except ValidationError as e:
            # Revert the change if validation fails
            setattr(scraper_config, key, original_value)
            print(f"\\033[1;31mError: Invalid value for {key}: {e}\\033[0m")
            return False

    except ValueError:
        error_msg = f"Invalid value type for {key}. Expected {expected_type.__name__}"
        print(f"\\033[1;31m{error_msg}\\033[0m")
        return False
    except json.JSONDecodeError:
        print(f"\\033[1;31mError: Invalid JSON format for list value for {key}\\033[0m")
        return False


def _normalize_url"""
    
    # Replace the function using regular expressions
    import re
    fixed_content = re.sub(old_function, new_function, content, flags=re.DOTALL)
    
    with open('wdbx_plugins/web_scraper.py', 'w', encoding='utf-8') as file:
        file.write(fixed_content)
    
    print("Fixed _update_config_value function in web_scraper.py")

if __name__ == "__main__":
    fix_update_config_function() 