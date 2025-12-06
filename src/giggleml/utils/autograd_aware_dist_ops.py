from typing import cast, override

import torch
import torch.distributed as dist
from torch.autograd.function import FunctionCtx

# INFO: ---------------------------
#         all_gather + cat
# ---------------------------------


class DifferentiableAllGatherCat(torch.autograd.Function):
    """
    Gathers tensors of varying sizes from all ranks and concatenates them.

    This is necessary, for example, when drop_last=False in a DistributedSampler.
    """

    @staticmethod
    @override
    def forward(
        ctx: FunctionCtx,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """
        Gathers tensors and saves all tensor sizes and rank for backward.
        """
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        # 1. GATHER SIZES
        # Get local batch size (dim 0)
        local_size_scalar = input_tensor.shape[0]

        # Create a 1-element tensor for the size
        local_size = torch.tensor(
            [local_size_scalar], dtype=torch.long, device=input_tensor.device
        )

        # Create a tensor to hold all sizes
        all_sizes = torch.empty(
            world_size, dtype=torch.long, device=input_tensor.device
        )

        # Gather all sizes
        dist.all_gather_into_tensor(all_sizes, local_size, group)

        # 2. GATHER DATA
        # Now create the tensor_list for the real gather
        tensor_list = [
            torch.empty(
                int(size.item()),
                *input_tensor.shape[1:],
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            for size in all_sizes
        ]

        # Perform the real all_gather
        dist.all_gather(tensor_list, input_tensor, group)

        # 3. CONCATENATE
        global_tensor = torch.cat(tensor_list, dim=0)

        # 4. SAVE FOR BACKWARD
        # We need all_sizes to know how to slice the gradient
        ctx.all_sizes = (  # pyright: ignore[reportAttributeAccessIssue]
            all_sizes.tolist()
        )  # [size_0, size_1, ...]
        ctx.rank = rank  # pyright: ignore[reportAttributeAccessIssue]
        ctx.process_group = group  # pyright: ignore[reportAttributeAccessIssue]
        return global_tensor

    @staticmethod
    @override
    def backward(ctx: FunctionCtx, *grad_outputs: torch.Tensor):
        """
        grad_output shape: (global_batch_size, embedding_dim)

        We need to sum all gradients and then slice the part
        corresponding to this rank.
        """
        assert len(grad_outputs) == 1
        grad_output = grad_outputs[0]

        # 1. SUM GRADIENTS
        group = ctx.process_group  # pyright: ignore[reportAttributeAccessIssue]
        dist.all_reduce(grad_output.contiguous(), op=dist.ReduceOp.SUM, group=group)

        # 2. DYNAMICALLY SLICE
        all_sizes = ctx.all_sizes  # pyright: ignore[reportAttributeAccessIssue]
        rank = ctx.rank  # pyright: ignore[reportAttributeAccessIssue]

        # Calculate the start and end index for this rank's slice
        start_idx = sum(all_sizes[:rank])
        end_idx = start_idx + all_sizes[rank]

        grad_input = grad_output[start_idx:end_idx]

        return grad_input, None  # the group doesn't need a grad


def all_gather_cat(
    local: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    return cast(torch.Tensor, DifferentiableAllGatherCat().apply(local, group))


# INFO: ----------------------------
#         all_reduce op=sum
# ----------------------------------


class DifferentiableAllReduceSum(torch.autograd.Function):
    """
    A custom autograd function for a distributed all_reduce (sum) operation.
    The backward pass is also an all_reduce (sum).
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """
        Forward pass:
        1. Clones the input tensor (since all_reduce is in-place).
        2. Performs all_reduce(sum) on the clone.
        3. Returns the all-reduced tensor.
        """
        # all_reduce is an in-place operation. We clone the input
        # to avoid modifying it in-place, which can cause issues.
        output_tensor = input_tensor.clone()

        # Perform the all-reduce operation
        dist.all_reduce(output_tensor, op=dist.ReduceOp.SUM, group=group)
        ctx.process_group = group  # pyright: ignore[reportAttributeAccessIssue]
        return output_tensor

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: torch.Tensor):
        """
        Backward pass:
        As derived above, the backward pass is also an all_reduce(sum)
        on the incoming gradients.
        """
        assert len(grad_outputs) == 1
        grad_output = grad_outputs[0]

        # Clone the grad_output for the in-place all_reduce
        grad_input = grad_output.clone()

        # Perform all_reduce(sum) on the gradients
        group = ctx.process_group  # pyright: ignore[reportAttributeAccessIssue]
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=group)

        # The function returns one gradient, corresponding to the
        # one input (input_tensor) of the forward pass.
        return grad_input, None


def all_reduce_sum(
    local: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    return cast(torch.Tensor, DifferentiableAllReduceSum().apply(local, group))
