"""Small notation helpers; callers supply every access and loop explicitly."""


def view(value, shape):
    axes = " ".join(f"(axis a{i} {n})" for i, n in enumerate(shape))
    return f"(view (tensor v{value}) (layout {axes}))"


def index(*parts):
    return "(keyed_index " + " ".join(f"(slot a{i} {p})" for i, p in enumerate(parts)) + ")"


def load(value, shape, access):
    return f"(load {view(value, shape)} {access})"


def store(value, shape, rhs, access):
    return f"(store {view(value, shape)} {rhs} {access})"


def all_gather(source, source_shape, source_index, destination, destination_shape, destination_index, axis):
    """Gather into rank-major regions; destination_index addresses rank zero's region."""
    return (
        f"(all_gather {view(source, source_shape)} {source_index} "
        f"{view(destination, destination_shape)} {destination_index} {axis})"
    )
