
def normalize_text(text:str) ->str:
    if isinstance(text, str): #evitando NaN
        return text.lower()
    return text