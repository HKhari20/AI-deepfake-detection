import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from detector import DeepfakeDetector

# Configuration
DATASET_PATH = "./dataset" # Structure should be: dataset/real/ and dataset/fake/
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def train_model():
    print("--- Initializing Deepfake Training Pipeline ---")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset folder '{DATASET_PATH}' not found.")
        print("Please create this folder and arrange your data like this:")
        print("  DeepfakeDetection/")
        print("  └── dataset/")
        print("      ├── real/  (Put authentic face frames here)")
        print("      └── fake/  (Put manipulated face frames here)")
        return

    # Transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Dataset (ImageFolder automatically assigns labels based on folder names)
    print("Loading dataset...")
    try:
        dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Found {len(dataset)} images belonging to {len(dataset.classes)} classes: {dataset.classes}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = DeepfakeDetector().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy for Fake/Real classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            # Match model output shape [batch_size, 1]
            labels = labels.float().unsqueeze(1).to(device) 

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(dataloader):.4f}")

    # Save the trained weights
    torch.save(model.state_dict(), "deepfake_model_weights.pth")
    print("Training complete! Weights saved to 'deepfake_model_weights.pth'")
    print("You can now load these weights in detector.py!")

if __name__ == "__main__":
    train_model()
