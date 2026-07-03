import json
import time
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import (
    DEVICE,
    LEARNING_RATE,
    PATIENCE,
    MODELS_DIR,
    METRICS_DIR)
class Trainer:

    def __init__(self, model, train_loader, val_loader, class_names, model_name):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.model_name = model_name
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.5)
        self.best_accuracy = 0.0
        self.counter = 0
        self.history = []
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, labels in self.train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        start = time.time()

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        inference_time = (time.time() - start) / len(self.val_loader.dataset) * 1000
        return total_loss / len(self.val_loader), accuracy, precision, recall, f1, inference_time

    def train(self, epochs):
        print(f"Начало обучения модели: {self.model_name}")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, acc, prec, rec, f1, inf_time = self.validate()
            self.scheduler.step()
            print(f"\nЭпоха {epoch + 1}/{epochs}")
            print(f"Loss (train): {train_loss:.4f}")
            print(f"Loss (val)  : {val_loss:.4f}")
            print(f"Accuracy    : {acc:.4f}")
            print(f"F1-score    : {f1:.4f}")
            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": acc,
                "f1": f1})

            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.counter = 0
                torch.save(
                    self.model.state_dict(),
                    MODELS_DIR / f"{self.model_name}_best.pth")
                print("Лучшая модель сохранена")

            else:
                self.counter += 1
                print(f"Нет улучшения ({self.counter}/{PATIENCE})")

                if self.counter >= PATIENCE:
                    print("Ранняя остановка обучения")
                    break

        with open(METRICS_DIR / f"{self.model_name}.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)

        print("Обучение завершено")