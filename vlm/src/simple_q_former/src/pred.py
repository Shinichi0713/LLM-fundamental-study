import torch
import matplotlib.pyplot as plt

model.eval()

max_test_steps = 3

for step, batch in enumerate(dataloader):
    if step >= max_test_steps:
        break

    # ========= データ =========
    pixel_values = batch["pixel_values"].to(device)

    # ========= Vision =========
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values)
        image_embeds = vision_outputs.last_hidden_state

        batch_size = pixel_values.size(0)

        # ========= Query tokens =========
        query_tokens = model.query_tokens.expand(batch_size, -1, -1)

        # ========= Q-Former =========
        qformer_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            return_dict=True
        )

        query_output = qformer_outputs.last_hidden_state

        # ========= Projection =========
        prefix_embeds = model.proj(query_output)

        # ========= T5 generate =========
        prefix_mask = torch.ones(
            batch_size,
            prefix_embeds.size(1),
            device=device
        )

        generated_ids = model.t5.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            max_length=30
        )

    # ========= decode =========
    texts = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    # ========= 出力 =========
    for i, text in enumerate(texts):
        print(f"\nStep {step} / Sample {i}")
        print("Generated:", text)

        # --- 画像表示 ---
        img = pixel_values[i].cpu().permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.show()
