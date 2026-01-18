device = "cuda" if torch.cuda.is_available() else "cpu"

vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

qformer = QFormer().to(device)

proj = nn.Linear(768, 768).to(device)
