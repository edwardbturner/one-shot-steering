# %%

import plotly.express as px  # type: ignore
import torch as t
from nnsight import LanguageModel  # type: ignore

from globals import MODEL_NAME, SEED
from utils import set_seed

# %%
set_seed(SEED)

model = LanguageModel(MODEL_NAME, device_map="auto", dispatch=True, torch_dtype=t.bfloat16)

tok = model.tokenizer

# %%


def get_qwen_n_layer_logit_lens(prompt: str, n: int):
    layers = model.model.layers
    probs_layers = []
    layer_indices = []  # Keep track of which layers we're using

    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                if layer_idx % n == 0:
                    # Process layer output through the model's head and layer normalization
                    layer_output = model.lm_head(model.model.norm(layer.output[0]))

                    # Apply softmax to obtain probabilities and save the result
                    probs = t.nn.functional.softmax(layer_output, dim=-1).save()  # type: ignore
                    probs_layers.append(probs)
                    layer_indices.append(layer_idx)  # Store the layer index

    probs = t.cat([probs.value for probs in probs_layers])

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)

    # Decode token IDs to words for each layer
    words = [
        [model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode() for t in layer_tokens]
        for layer_tokens in tokens
    ]

    # Access the 'input_ids' attribute of the invoker object to get the input words
    input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]

    max_probs_float = max_probs.float()

    fig = px.imshow(
        max_probs_float.detach().cpu().numpy(),
        x=input_words,
        y=layer_indices,  # Use actual layer indices
        color_continuous_scale=px.colors.diverging.RdYlBu_r,
        color_continuous_midpoint=0.50,
        text_auto=True,
        labels=dict(x="Input Tokens", y="Layer Number", color="Probability"),
    )

    fig.update_layout(
        title=f"Logit Lens Visualization for every {n} layers",
        xaxis_tickangle=0,
        font=dict(size=12),  # Increase base font size
    )

    # Update text size and format
    fig.update_traces(
        text=words,
        texttemplate="%{text}",
        textfont=dict(size=10),  # Increase text size in cells
    )

    # Adjust cell size for better readability
    fig.update_layout(
        width=800,  # Increase figure width
        height=600,  # Increase figure height
    )

    fig.show()


# %%

get_qwen_n_layer_logit_lens("The Eiffel Tower is in the city of", 4)


# %%
