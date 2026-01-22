from transformers import AutoModel

model = AutoModel.from_pretrained("./models/base/vjepa2")

print("Model architecture:")
print("="*70)
for name, module in model.named_modules():
    if 'attn' in name.lower() or 'proj' in name.lower() or 'query' in name.lower() or 'key' in name.lower() or 'value' in name.lower():
        print(f"{name}: {type(module).__name__}")