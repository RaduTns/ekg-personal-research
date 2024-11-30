from models.model_dataset import ECGDataset
from scripts import process_data, helper_functions
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from training_models.model_structure import load_inception_v3, load_resnet18
from training_models.train_model import train_model, test_model

folder_path = '..\\data\\processed_data\\2nd_try'
coordinates = (67, 282, 2177, 1518)
# Define data paths and hyperparameters
data_dir = folder_path
num_classes = 4  # Adjust based on your dataset
batch_size = 32
learning_rate = 0.05
num_epochs = 10

# Split data into training, validation, and test sets (70%, 20%, 10%)

train_size = 0.7
val_size = 0.2
test_size = 0.1


def main():
    print()
    dataset = ECGDataset(data_dir)
    train_len, val_len = int(len(dataset) * train_size), int(len(dataset) * val_size)
    test_len = len(dataset) - train_len - val_len
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

    # Create DataLoaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No shuffling for validation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No shuffling for testing

    model = load_resnet18(num_classes=num_classes)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Categorical cross-entropy for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Train the model
    train_model(model, train_loader, val_loader, optimizer, criterion, num_classes, num_epochs=num_epochs)
    test_model(model, test_loader, criterion, num_classes=num_classes)

    # Save the trained model for future use
    torch.save(model.state_dict(), "..\\training_models\\inception_v3.pth")

if __name__ == "__main__":
    main()