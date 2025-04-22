from typing import Optional, Tuple, Union

import torch as t
from nnsight import Envoy, LanguageModel  # type: ignore
from tqdm import tqdm


@t.no_grad()
def get_log_probs(
    model: LanguageModel,
    prompt: str,
    completion: str,
    coldness: float,
) -> t.Tensor:
    """
    Get the log probabilities of the completion tokens.

    Args:
        model: The model to use.
        prompt: The prompt to use.
        completion: The completion to use.
        coldness: The coldness to use (inverse temperature).

    Returns:
        The log probabilities of the completion tokens.
    """
    tok = model.tokenizer
    eps = t.finfo(model.dtype).eps
    prompt_tokens = tok.encode(prompt)
    all_tokens = tok.encode(prompt + completion)
    completion_tokens = all_tokens[len(prompt_tokens) :]

    # Compute token probabilities
    logits = model.trace(all_tokens, trace=False).logits

    # Scale logits by inverse temperature
    probs = t.nn.functional.softmax(logits * coldness, dim=-1)

    current_loss = t.tensor(0.0, dtype=model.dtype, device=model.device)
    for idx, completion_token in enumerate(completion_tokens):
        # Compute index in whole sequence, then get probability of target
        prompt_token_idx = len(prompt_tokens) + idx - 1
        target_prob = probs[0, prompt_token_idx, completion_token]

        # Compute log probability
        target_prob = t.log(target_prob + eps)

        # Add to loss
        current_loss += target_prob

    return current_loss


@t.no_grad()
def get_steered_log_probs(
    model: LanguageModel,
    prompt: str,
    completion: str,
    coldness: float,
    vector: t.Tensor,
    module: Envoy,
) -> t.Tensor:
    """
    Get the log probabilities of the completion tokens given a steering vector.

    Args:
        model: The model to use.
        prompt: The prompt to use.
        completion: The completion to use.
        coldness: The coldness to use (inverse temperature).
        vector: The steering vector to use.
        module: The module to use.

    Returns:
        The log probabilities of the steered completion tokens.
    """
    tok = model.tokenizer
    eps = t.finfo(model.dtype).eps
    prompt_tokens = tok.encode(prompt)
    all_tokens = tok.encode(prompt + completion)
    completion_tokens = all_tokens[len(prompt_tokens) :]

    # Compute token probabilities
    with model.trace(all_tokens):
        x = module.output[0]
        x_hat = x + vector
        module.output = (x_hat,)
        logits = model.output.logits.save()

    # Scale logits by inverse temperature
    probs = t.nn.functional.softmax(logits * coldness, dim=-1)

    current_loss = t.tensor(0.0, dtype=model.dtype, device=model.device)
    for idx, completion_token in enumerate(completion_tokens):
        # Compute index in whole sequence, then get probability of target
        prompt_token_idx = len(prompt_tokens) + idx - 1
        target_prob = probs[0, prompt_token_idx, completion_token]

        # Compute log probability
        target_prob = t.log(target_prob + eps)

        # Add to loss
        current_loss += target_prob

    return current_loss


def _get_steering_loss(
    model: LanguageModel,
    prompt_tokens: t.Tensor,
    completion_tokens: t.Tensor,
    coldness: float,
    vector: t.Tensor,
    module: Envoy,
) -> t.Tensor:
    """
    Compute the completion loss given an intervention.

    Args:
        model: The model to use.
        prompt_tokens: The prompt tokens to use.
        completion_tokens: The completion tokens to use.
        coldness: The coldness to use (inverse temperature).
        vector: The steering vector to use.
        module: The module to use.

    Returns:
        The completion loss.
    """
    eps = t.finfo(model.dtype).eps
    # NOTE: Assumes steering at layer outputs
    with model.trace(prompt_tokens + completion_tokens):
        x = module.output[0]
        x_hat = x + vector
        module.output = (x_hat,)
        logits = model.output.logits.save()

    # Scale logits by inverse temperature
    probs = t.nn.functional.softmax(logits * coldness, dim=-1)

    current_loss = t.tensor(0.0, dtype=model.dtype, device=model.device)
    for idx, completion_token in enumerate(completion_tokens):
        # Compute index in whole sequence
        prompt_token_idx = len(prompt_tokens) + idx - 1

        # Compute the log probability of the target token
        target_prob = probs[0, prompt_token_idx, completion_token]
        target_prob = t.log(target_prob + eps)

        # Add to loss
        current_loss -= target_prob

        # NOTE: Uncomment for debugging
        # print(
        #     idx,
        #     target_prob.item(),
        #     completion_token,
        # )

    return current_loss


@t.no_grad()
def _normalize_vector(vector: t.Tensor, max_norm: Optional[float] = None) -> None:
    """
    Normalize the vector to have a maximum norm.

    Args:
        vector: The vector to normalize.
        max_norm: The maximum norm to use.
    """
    if max_norm is not None:
        current_norm = vector.norm()

        if current_norm > max_norm:
            vector[:] = max_norm * vector / current_norm


# Currently only supports single prompt -> completion training
def optimize_vec(
    model: LanguageModel,
    prompt: str,
    completion: str,
    module: Envoy,
    lr: float,
    max_iters: int,
    coldness: float,
    target_loss: float,
    d_model: int,
    max_norm: Optional[float] = None,
    satisfice: bool = False,
    return_loss: bool = False,
) -> Union[t.Tensor, Tuple[t.Tensor, float]]:
    """
    Optimize the vector to steer the completion.

    Args:
        model: The model to use.
        prompt: The prompt to use.
        completion: The completion to use.
        module: The module to use.
        lr: The learning rate to use.
        max_iters: The maximum number of iterations to use.
        coldness: The coldness to use (inverse temperature).
        target_loss: The target loss to use.
        d_model: The dimension of the model.
        max_norm: The maximum norm to use.
        satisfice: Whether to satisfice.
        return_loss: Whether to return the loss.

    Returns:
        The optimized vector.
    """
    # Tokenize prompt and completion
    tok = model.tokenizer
    prompt_tokens = tok.encode(prompt)
    all_tokens = tok.encode(prompt + completion)
    completion_tokens = all_tokens[len(prompt_tokens) :]

    # Check that .encode isn't batching
    assert len(all_tokens) > 1

    vector = t.randn(d_model, dtype=model.dtype, device=model.device)
    vector.requires_grad_(True)
    optim = t.optim.Adam([vector], lr=lr)

    eps = t.finfo(model.dtype).eps
    target_loss = target_loss if not satisfice else target_loss + eps

    pbar = tqdm(range(max_iters))
    for _ in pbar:
        optim.zero_grad()
        current_loss = _get_steering_loss(model, prompt_tokens, completion_tokens, coldness, vector, module)

        if satisfice:
            current_loss = (current_loss - target_loss).pow(2)

        current_loss.backward()
        optim.step()
        pbar.set_postfix(loss=current_loss.item())

        _normalize_vector(vector, max_norm=max_norm)

        if current_loss <= target_loss:
            break

    if return_loss:
        return vector, current_loss.item()

    return vector
