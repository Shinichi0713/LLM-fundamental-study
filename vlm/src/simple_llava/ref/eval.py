@torch.no_grad()
def generate_caption(
    image_path,
    vision_encoder,
    qformer,
    query_tokens,
    proj,
    llm,
    processor,
    tokenizer,
    device,
    max_new_tokens=30,
):
    vision_encoder.eval()
    qformer.eval()
    llm.eval()

    # --- Image ---
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    )["pixel_values"].to(device)

    # --- Vision Encoder ---
    vision_feats = vision_encoder(
        pixel_values
    ).last_hidden_state

    # --- Q-Former ---
    query_embeds = query_tokens.expand(1, -1, -1)

    q_out = qformer(
        query_embeds=query_embeds,
        encoder_hidden_states=vision_feats,
    ).last_hidden_state

    img_tokens = proj(q_out)   # [1, num_query, n_embd]

    # --- BOS token ---
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    bos = torch.tensor([[bos_id]], device=device)

    bos_embed = llm.transformer.wte(bos)

    # --- concat ---
    inputs_embeds = torch.cat(
        [img_tokens, bos_embed],
        dim=1
    )

    # --- generate ---
    outputs = llm.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
