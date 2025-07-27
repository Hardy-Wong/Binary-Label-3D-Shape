import torch


def select_positions(
    input_array: torch.Tensor, num_of_f_in_1_node: int = 12
) -> torch.Tensor:
    """select only position vectors. shape (B, W)

    Args:
        input_array (torch.Tensor): input array which contains both positions and other features.
        num_of_f_in_1_node (int, optional): number of features in 1 node. Defaults to 12.

    Returns:
        torch.Tensor: position vectors. shape (B, 3*num_nodes)
    """
    indices = torch.tensor(list(range(3)))
    batch_size = len(input_array)
    input_positions = torch.index_select(
        input_array.reshape((batch_size, -1, num_of_f_in_1_node)), 2, indices
    )
    return input_positions.reshape((batch_size, -1))
