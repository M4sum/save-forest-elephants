from src.data.api import Event


def event_overlaps_with_window(event: Event, window_start: float, window_end: float) -> bool:
    return is_between(event[0], window_start, window_end) or is_between(event[1], window_start, window_end) \
            or is_between(window_start, event[0], event[1]) or is_between(window_end, event[0], event[1])

def is_between(a, min_value, max_value):
    return min_value <= a <= max_value


def fit_event_to_window(event: Event, window_start: float, window_end: float) -> Event:
    return max(event[0], window_start), min(event[1], window_end)




