def clone_net_structure(net_structure):
    """Clones list of keras layers

    Parameters:
    ----------
    net_structure: list
        sequence of layers to clone

    Returns
    ----------
    new_net_structure: list
        sequence of cloned layers
    """
    new_net_structure = []
    for layer in net_structure:
        new_layer = layer.__class__.from_config(layer.get_config())
        new_net_structure.append(new_layer)
    return new_net_structure