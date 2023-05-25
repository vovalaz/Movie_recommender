import torch
import torch.nn as nn
import torch.optim as optim

# Sample user-item matrix (rows: users, columns: items)
user_item_matrix = torch.tensor([
    [5, 3, 0, 1, 4, 0],
    [0, 0, 0, 4, 0, 0],
    [4, 0, 0, 2, 0, 0],
    [0, 1, 5, 4, 3, 0],
    [0, 0, 4, 0, 0, 5],
    [0, 0, 4, 0, 0, 0],
    [0, 0, 0, 4, 0, 0]
], dtype=torch.float)

# Sample item features matrix (rows: items, columns: features)
item_features_matrix = torch.tensor([
    [0.80, 0.50, 0.20],
    [0.60, 0.70, 0.80],
    [0.90, 0.40, 0.10],
    [0.30, 0.60, 0.90],
    [0.70, 0.20, 0.30],
    [0.40, 0.90, 0.60]
], dtype=torch.float)


# Collaborative filtering model
class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(CollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        user_item_embedded = torch.mul(user_embedded, item_embedded)
        predicted_ratings = self.fc(user_item_embedded).squeeze()
        return predicted_ratings


# Content-based filtering model
class ContentBasedFiltering(nn.Module):
    def __init__(self, num_items, num_features, embedding_dim):
        super(ContentBasedFiltering, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, item_indices):
        item_embedded = self.item_embedding(item_indices)
        predicted_ratings = self.fc(item_embedded).squeeze()
        return predicted_ratings


# Recommendation system combining collaborative and content-based filtering
class RecommendationSystem(nn.Module):
    def __init__(self, num_users, num_items, num_features, embedding_dim):
        super(RecommendationSystem, self).__init__()
        self.collaborative_model = CollaborativeFiltering(num_users, num_items, embedding_dim)
        self.content_model = ContentBasedFiltering(num_items, num_features, embedding_dim)

    def forward(self, user_indices, item_indices):
        collaborative_ratings = self.collaborative_model(user_indices, item_indices)
        content_ratings = self.content_model(item_indices)
        combined_ratings = collaborative_ratings + content_ratings
        return combined_ratings


# Training loop
def train_recommendation_system(model, user_item_matrix, item_features_matrix, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        user_indices, item_indices = torch.where(user_item_matrix != 0)
        predicted_ratings = model(user_indices, item_indices)
        true_ratings = user_item_matrix[user_indices, item_indices]
        loss = loss_fn(predicted_ratings, true_ratings)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


# Generate recommendations
def generate_recommendations(user_id, user_item_matrix, item_features_matrix, model, top_n=5):
    user_indices = torch.tensor([user_id] * item_features_matrix.size(0))
    item_indices = torch.tensor(range(item_features_matrix.size(0)))
    predicted_ratings = model(user_indices, item_indices)
    _, top_indices = torch.topk(predicted_ratings, top_n)
    recommendations = top_indices.tolist()
    return recommendations


# Example usage
num_users = user_item_matrix.size(0)
num_items = user_item_matrix.size(1)
num_features = item_features_matrix.size(1)
embedding_dim = 10
num_epochs = 100
learning_rate = 0.01

model = RecommendationSystem(num_users, num_items, num_features, embedding_dim)
train_recommendation_system(model, user_item_matrix, item_features_matrix, num_epochs, learning_rate)

user_id = 0
recommendations = generate_recommendations(user_id, user_item_matrix, item_features_matrix, model)
print(f"Recommendations for user {user_id}: {recommendations}")
