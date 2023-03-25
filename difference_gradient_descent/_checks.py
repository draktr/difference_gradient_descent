from optschedule import Schedule


def _check_iterables(differences, learning_rates, epochs):
    if not isinstance(epochs, int):
        raise ValueError("Number of epochs should be an integer")
    if isinstance(differences, (int, float)):
        differences = Schedule().constant(differences[0])
    if isinstance(learning_rates, (int, float)):
        learning_rates = Schedule(epochs).constant(learning_rates[0])
    if differences.shape[0] != learning_rates.shape[0]:
        raise ValueError("Number of differences and learning rates should be equal.")
    if epochs != differences.shape[0]:
        raise ValueError(
            "Number of epochs, differences and learning rates given should be equal."
        )

    return differences, learning_rates, epochs
