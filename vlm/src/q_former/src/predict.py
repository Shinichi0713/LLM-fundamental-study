from PIL import Image
import requests
from torchvision import transforms

img = Image.open(requests.get("https://msp.c.yimg.jp/images/v2/FUTi93tXq405grZVGgDqG7G_oA9zadA5ADwFCE1YU3tdLUonL58QM1KfOuDXHsZ5XDvlvwCFkcrUgkQLne609_Sv784t1OdNJS_OtUCIEomWAyAuolk6_I_WVGo_Gj1vvoar5Njt-54G7l38StZ0KUZfeDnJ2nWj_Iv9KRQ-FPoqQt_bAO5vfdBDEnW3gP6g4BT44Vdk54c-85ajfLcZ_4JHEyAYS-tFUUCGEwy-3NRqQegVFhWc64IcB9YbRJ4n/haircolor1.jpg?errorImage=false", stream=True).raw)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    vision_out = vision_encoder(x).last_hidden_state
    q_out = qformer(vision_out)
    llm_in = proj(q_out)

    prompt = "Describe the image: "
    tokens = tokenizer(prompt, return_tensors="pt").to(device)

    # T5のEncoderへ inputs_embeds を渡す
    encoder_inputs = torch.cat(
        [llm_in, llm.shared(tokens.input_ids)],
        dim=1
    )

    enc_out = llm.encoder(inputs_embeds=encoder_inputs)

    out = llm.generate(
        encoder_outputs=enc_out,
        max_length=48
    )

    caption = tokenizer.decode(out[0], skip_special_tokens=True)

print("Caption:", caption)

