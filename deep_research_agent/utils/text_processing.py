from difflib import SequenceMatcher

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text segments using SequenceMatcher.
    
    Args:
        text1: First text segment
        text2: Second text segment
        
    Returns:
        float: Similarity score between 0 and 1
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() 