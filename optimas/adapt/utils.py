
def format_input_fields(**input_fields):
    """
    Formats input fields into a string representation.
    
    Args:
        **input_fields: Keyword arguments representing the input fields.
        
    Returns:
        str: A formatted string of input fields.
    """
    return "\n".join(f"{key}: \n  {value}" for key, value in input_fields.items())

