import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score

class Seq2Seq(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, seq_len):
        super(Seq2Seq, self).__init__()
        self.seq_len = seq_len
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # Repeat to fit the seq_len dimension
        h, _ = self.rnn(x)  # LSTM output
        out = self.fc(h)  # Linear layer
        return out

class SequenceDataset(Dataset):
    def __init__(self, embeddings, ids):
        self.embeddings = embeddings
        self.ids = ids
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.ids[idx]

def exact_match_rate(y_true, y_pred):
    correct = (y_true == y_pred).all(dim=1)
    return correct.float().mean().item()

def topk_exact_match_rate(y_true, topk_preds):
    # 对于每一行，检查 true 值是否在 top-5 预测中
    correct = torch.any(topk_preds == y_true.unsqueeze(-1), dim=-1)
    return correct.float().mean().item()

def topk_accuracy(y_true, y_pred, k):
    _, topk_pred = y_pred.topk(k, dim=1, largest=True, sorted=False)
    correct = torch.any(topk_pred == y_true.unsqueeze(1), dim=1)
    return correct.float().mean().item()

def evaluation(seq_embedding, task_data, num_nodes, k=1, fold=5, acc_at = 1):
    # Convert labels to tensor
    task_data['pre_label_tensor'] = task_data['pre_label'].apply(lambda x: torch.tensor(x))
    y = torch.stack(task_data['pre_label_tensor'].tolist())
    dataset = SequenceDataset(seq_embedding, y)

    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    seed = 488
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(seq_embedding.shape[-1], 128, num_nodes, k).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    num_epochs = 120

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_embeddings, batch_ids in train_loader:
            batch_embeddings, batch_ids = batch_embeddings.to(device), batch_ids.to(device)
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            outputs = outputs.view(-1, num_nodes)
            batch_ids = batch_ids.view(-1)
            loss = criterion(outputs, batch_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

        # Evaluate the model
        model.eval()
        all_y_true = []
        all_y_pred = []
        with torch.no_grad():
            for batch_embeddings, batch_ids in test_loader:
                batch_embeddings, batch_ids = batch_embeddings.to(device), batch_ids.to(device)
                outputs = model(batch_embeddings)
                # outputs: [batch_size, seq_len, num_classes]
                all_y_true.append(batch_ids.cpu())
                all_y_pred.append(outputs.cpu())

        all_y_true = torch.cat(all_y_true)
        all_y_pred = torch.cat(all_y_pred)

        # Convert outputs to probabilities
        all_y_pred = torch.softmax(all_y_pred, dim=-1)
        if acc_at > 1:
            # 获取前5个预测的索引
            topk_preds = torch.topk(all_y_pred, acc_at, dim=-1).indices

            # 计算 top-5 的匹配率
            topk_match = topk_exact_match_rate(all_y_true, topk_preds)
            print(f"Epoch {epoch+1}/{num_epochs},Top-5 Exact Match Rate: {topk_match:.4f}")
        else:
            # Calculate metrics
            exact_match = exact_match_rate(all_y_true, all_y_pred.argmax(dim=-1))
            # top_k_accuracy = topk_accuracy(all_y_true, all_y_pred, k)
            print(f'Epoch {epoch+1}/{num_epochs}, Exact Match Rate: {exact_match:.4f}')



        




       
