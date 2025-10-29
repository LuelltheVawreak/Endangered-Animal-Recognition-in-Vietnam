import torch
import clip
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


CLIP_MODEL_NAME = "ViT-B/16"
MODEL_PATH = "biotroveclip-vit-b-16-from-openai-epoch-40.pt"
BATCH_SIZE = 2
EPOCHS = 10
LR = 2e-5


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
epoch_loss, epoch_top1_acc, epoch_top3_acc = [], [], []

# --- data augmentation ---
augment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# --- fine-tune ---
def fine_tune_clip(dataset_path, epochs=10, batch_size=2, lr=2e-5):
    print("Start fine-tuning on dataset:", dataset_path)

    for p in model.parameters():
        p.requires_grad = False

    train_dataset = datasets.ImageFolder(dataset_path, transform=augment)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_classes = len(train_dataset.classes)
    print("Number of classes:", num_classes, train_dataset.classes)

    classifier = nn.Linear(model.visual.output_dim, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        all_logits = []
        all_targets = []

        for images, targets in train_loader:
            images = images.to(device)
            # Assuming texts are generated from targets or dataset provides them
            texts = [f"a photo of a {train_dataset.classes[target]}" for target in targets] # Example text generation
            texts = clip.tokenize(texts, truncate=True).to(device)

            optimizer.zero_grad()

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # normalize
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            loss_i = F.cross_entropy(logits_per_image, ground_truth)
            loss_t = F.cross_entropy(logits_per_text, ground_truth)
            loss = (loss_i + loss_t) / 2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_logits.append(logits_per_image.detach().cpu())
            all_targets.append(ground_truth.detach().cpu())

        # --- avr loss ---
        avg_loss = running_loss / len(train_loader)

        # ---  top-1  top-3 accuracy ---
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        probs = F.softmax(all_logits, dim=1)
        topk = torch.topk(probs, k=3, dim=1).indices
        correct1 = (topk[:, :1] == all_targets.unsqueeze(1)).any(dim=1).sum().item()
        correct3 = (topk == all_targets.unsqueeze(1)).any(dim=1).sum().item()
        top1 = 100.0 * correct1 / all_targets.size(0)
        top3 = 100.0 * correct3 / all_targets.size(0)

        # ---save log ---
    log_df = pd.DataFrame({
        "epoch": list(range(1, EPOCHS+1)),
        "loss": epoch_loss,
        "top1_acc": epoch_top1_acc,
        "top3_acc": epoch_top3_acc
    })
    log_df.to_csv("training_log.csv", index=False)
    print(" Log đã được lưu vào training_log.csv")

   
    plt.figure(figsize=(10,5))
    plt.plot(log_df["epoch"], log_df["loss"], label="Loss")
    plt.plot(log_df["epoch"], log_df["top1_acc"], label="Top-1 Acc")
    plt.plot(log_df["epoch"], log_df["top3_acc"], label="Top-3 Acc")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training progress")
    plt.show()
