# Get murmur from patient data.
def get_murmur(data):
    murmur = None
    for text in data.split("\n"):
        if text.startswith("#Murmur:"):
            murmur = text.split(": ")[1]
    if murmur is None:
        raise ValueError(
            "No murmur available. Is your code trying to load labels from the hidden data?"
        )
    return murmur


# Get outcome from patient data.
def get_outcome(data):
    outcome = None
    for text in data.split("\n"):
        if text.startswith("#Outcome:"):
            outcome = text.split(": ")[1]
    if outcome is None:
        raise ValueError(
            "No outcome available. Is your code trying to load labels from the hidden data?"
        )
    return outcome
