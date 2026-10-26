"""Keep requested results across reruns, invalidating them when inputs change."""


def requested(state, key, context, clicked):
    if clicked:
        state[key] = {"context": context}
    elif state.get(key, {}).get("context") != context:
        state.pop(key, None)
    return key in state


def result(state, key, compute):
    entry = state[key]
    if "value" not in entry:
        entry["value"] = compute()
    return entry["value"]
