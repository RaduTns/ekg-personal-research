from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, criterion, num_classes, num_epochs, device="cpu"):
  model.to(device)
  # Training loop
  for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train loop (set model to train mode)
    model.train()
    running_loss = 0.0
    correct_predictions = 0.0
    total_predictions = 0.0
    true_positives = torch.zeros(num_classes, device=device)  # Initialize for all classes
    false_positives = torch.zeros(num_classes, device=device)
    false_negatives = torch.zeros(num_classes, device=device)

    # Wrap train_loader with tqdm for progress bar
    train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    for images, labels in train_loader_iter:
      images = images.to(device)  # Move only images to device
      labels = labels.to(device)  # Move only labels to device

      # Clear gradients from previous iteration
      optimizer.zero_grad()

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)  # Calculate loss

      # Backward pass and parameter update
      loss.backward()
      optimizer.step()

      # Update running loss
      running_loss += loss.item()

      # Calculate accuracy
      _, predicted = torch.max(outputs.data, 1)
      correct_predictions += (predicted == labels).sum().item()
      total_predictions += len(labels)

      # Update class-wise statistics for precision, recall, F1
      for i in range(len(labels)):
        predicted_label = predicted[i]
        true_label = labels[i]
        if predicted_label == true_label:
          true_positives[true_label] += 1  # True positive for the class
        else:
          false_positives[predicted_label] += 1  # False positive for predicted class
          false_negatives[true_label] += 1  # False negative for true class

      # Update tqdm description
      train_loader_iter.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_predictions
    print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Calculate precision, recall, F1 score (per class and overall)
    precision = torch.zeros(num_classes, device=device)
    recall = torch.zeros(num_classes, device=device)
    for class_index in range(num_classes):
      if true_positives[class_index] > 0:
        precision[class_index] = true_positives[class_index] / (
                  true_positives[class_index] + false_positives[class_index])
      if true_positives[class_index] > 0:
        recall[class_index] = true_positives[class_index] / (true_positives[class_index] + false_negatives[class_index])
      if precision[class_index] > 0 and recall[class_index] > 0:
        f1 = 2 * precision[class_index] * recall[class_index] / (precision[class_index] + recall[class_index])
        print(
          f"Class {class_index} - Precision: {precision[class_index]:.4f}, Recall: {recall[class_index]:.4f}, F1: {f1:.4f}")



def test_model(model, test_loader, criterion, num_classes, device="cpu"):
      model.to(device)
      model.eval()  # Set model to evaluation mode
      running_loss = 0.0
      correct_predictions = 0.0
      total_predictions = 0.0
      all_labels = []
      all_predictions = []

      true_positives = torch.zeros(num_classes, device=device)  # Initialize for all classes
      false_positives = torch.zeros(num_classes, device=device)
      false_negatives = torch.zeros(num_classes, device=device)

      # Wrap test_loader with tqdm for progress bar
      test_loader_iter = tqdm(test_loader, desc="Testing", unit="batch")
      with torch.no_grad():  # Disable gradient computation for testing
        for images, labels in test_loader_iter:
          images = images.to(device)
          labels = labels.to(device)

          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)

          # Update running loss
          running_loss += loss.item()

          # Calculate accuracy
          _, predicted = torch.max(outputs.data, 1)
          correct_predictions += (predicted == labels).sum().item()
          total_predictions += len(labels)

          all_labels.extend(labels.cpu().numpy())
          all_predictions.extend(predicted.cpu().numpy())

          # Update class-wise statistics for precision, recall, F1
          for i in range(len(labels)):
            predicted_label = predicted[i]
            true_label = labels[i]
            if predicted_label == true_label:
              true_positives[true_label] += 1  # True positive for the class
            else:
              false_positives[predicted_label] += 1  # False positive for predicted class
              false_negatives[true_label] += 1  # False negative for true class

      test_loss = running_loss / len(test_loader)
      test_accuracy = correct_predictions / total_predictions
      print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

      # Calculate precision, recall, F1 score (per class and overall)
      precision = torch.zeros(num_classes, device=device)
      recall = torch.zeros(num_classes, device=device)
      f1_scores = torch.zeros(num_classes, device=device)

      for class_index in range(num_classes):
        if true_positives[class_index] > 0:
          precision[class_index] = true_positives[class_index] / (
                  true_positives[class_index] + false_positives[class_index])
        if true_positives[class_index] > 0:
          recall[class_index] = true_positives[class_index] / (
                  true_positives[class_index] + false_negatives[class_index])
        if precision[class_index] > 0 and recall[class_index] > 0:
          f1_scores[class_index] = 2 * precision[class_index] * recall[class_index] / (
                    precision[class_index] + recall[class_index])

      for class_index in range(num_classes):
        print(
          f"Class {class_index} - Precision: {precision[class_index]:.4f}, Recall: {recall[class_index]:.4f}, F1: {f1_scores[class_index]:.4f}")

      # Overall metrics
      overall_precision = precision_score(all_labels, all_predictions, average='weighted')
      overall_recall = recall_score(all_labels, all_predictions, average='weighted')
      overall_f1 = f1_score(all_labels, all_predictions, average='weighted')
      print(
        f"Overall Precision: {overall_precision:.4f}, Overall Recall: {overall_recall:.4f}, Overall F1: {overall_f1:.4f}")

    # Overall