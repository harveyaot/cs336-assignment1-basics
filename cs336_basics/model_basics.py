import torch
import einx
from einops import rearrange, einsum


channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32 * 32, 32 * 32)
## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
print("channels_last_flat.shape using view: ", channels_last_flat.shape)
# channels_last_flat.shape using view:  torch.Size([64, 1024, 3])

channels_first_flat = channels_last_flat.transpose(1, 2)
print("channels_first_flat.shape using transpose: ", channels_first_flat.shape)
# channels_first_flat.shape using transpose:  torch.Size([64, 3, 1024])


channels_first_flat_transformed = channels_first_flat @ B.T
print(
    "channels_first_flat_transformed.shape using @: ",
    channels_first_flat_transformed.shape,
)
# channels_first_flat_transformed.shape using @:  torch.Size([64, 3, 1024])

channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
print(
    "channels_last_flat_transformed.shape using transpose: ",
    channels_last_flat_transformed.shape,
)
# channels_last_flat_transformed.shape using transpose:  torch.Size([64, 1024, 3])

channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
print("channels_last_transformed.shape using view: ", channels_last_transformed.shape)
# channels_last_transformed.shape using view:  torch.Size([64, 32, 32, 3])


height = width = 32
## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
    channels_last, "batch height width channel -> batch channel (height width)"
)
print("channels_first.shape using rearrange: ", channels_first.shape)
# channels_first.shape using rearrange:  torch.Size([64, 3, 1024])

channels_first_transformed = einsum(
    channels_first,
    B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out",
)
print("channels_last_transformed.shape using view: ", channels_last_transformed.shape)
# channels_last_transformed.shape using view:  torch.Size([64, 32, 32, 3])

channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height,
    width=width,
)
print(
    "channels_last_transformed.shape using rearrange: ", channels_last_transformed.shape
)
# channels_last_transformed.shape using view:  torch.Size([64, 32, 32, 3])

height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last,
    B,
    col_in=width,
    col_out=width,
)
