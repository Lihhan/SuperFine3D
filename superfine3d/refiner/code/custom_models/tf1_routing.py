"""Restrict coordinate-based MSGA routes to immutable spatial macroblocks."""
import torch


def bound_routes(coords, routes, block_size):
    if block_size < 1:
        raise ValueError('Positive latent block edge required')
    blocks = torch.div(coords[:, 1:].long(), block_size, rounding_mode='floor')
    result = []
    for batch, query, keys in routes:
        # Shifted cells can straddle a macroblock. Split queries as well as keys.
        unique, inverse = torch.unique(blocks[query], dim=0, return_inverse=True)
        for group, block in enumerate(unique):
            qi = query[inverse == group]
            ki = keys[(blocks[keys] == block).all(1)]
            # Ensure complete intra-cell connectivity, without duplicate softmax mass.
            ki = torch.unique(torch.cat((ki, qi)), sorted=True)
            result.append((batch, qi, ki))
    return result


def lr_self_post(block, source, attention_output, gate, shift, scale, mlp_gate):
    """Clean LR memory gets self-attention and FFN; no image or HR cross-attention."""
    hidden = source + attention_output * gate
    mlp_input = hidden.replace(block.norm3(hidden.feats)) * (1 + scale) + shift
    return hidden + block.mlp(mlp_input) * mlp_gate
