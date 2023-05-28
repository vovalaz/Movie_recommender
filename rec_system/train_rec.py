import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(RecommendationModel, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_content_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_content_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, embedding_dim)

    def forward(self, user_indices, item_indices):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        cf_output = torch.sum(user_embed * item_embed, dim=1)

        user_content_embed = self.user_content_embedding(user_indices)
        item_content_embed = self.item_content_embedding(item_indices)
        cb_input = torch.cat((user_content_embed, item_content_embed), dim=1)
        cb_output = F.relu(self.fc1(cb_input))
        cb_output = self.fc2(cb_output)

        output = cf_output + cb_output

        return output


# Load movie details from CSV
movies_df = pd.read_csv("movies.csv")
num_movies = len(movies_df)

# Load user movie ratings from CSV
ratings_df = pd.read_csv("ratings.csv")
num_users = ratings_df["user_id"].nunique()

# Map movie and user IDs to sequential indices
movies_df["movie_index"] = pd.factorize(movies_df["movie_id"])[0]
ratings_df["user_index"] = pd.factorize(ratings_df["user_id"])[0]
ratings_df["movie_index"] = pd.factorize(ratings_df["movie_id"])[0]

# Prepare the data for training
user_indices = torch.tensor(ratings_df["user_index"].values, dtype=torch.long)
item_indices = torch.tensor(ratings_df["movie_index"].values, dtype=torch.long)
ratings = torch.tensor(ratings_df["rating"].values, dtype=torch.float32)

# Initialize the model
embedding_dim = 128
model = RecommendationModel(num_users, num_movies, embedding_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    outputs = model(user_indices, item_indices)

    # Compute the loss
    loss = criterion(outputs, ratings)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print the loss for every epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
