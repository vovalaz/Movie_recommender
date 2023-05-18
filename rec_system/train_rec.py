import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Load the movie dataset
ratings_data = pd.read_csv("movie_datasets/ratings_small.csv")
movies_data = pd.read_csv("movie_datasets/movies_metadata.csv", low_memory=False)
movies_data.rename(columns={"id": "movieId"}, inplace=True)

# Merge ratings and movies data
merged_data = pd.merge(ratings_data, movies_data, on="movieId")

# Split the data into train and test sets
train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)

# Create user and movie dictionaries
user_dict = {uid: i for i, uid in enumerate(merged_data["userId"].unique())}
movie_dict = {mid: i for i, mid in enumerate(merged_data["movieId"].unique())}

# Number of users and movies
num_users = len(user_dict)
num_movies = len(movie_dict)


# Define the dataset class
class MovieLensDataset(Dataset):
    def __init__(self, data, user_dict, movie_dict):
        self.data = data
        self.user_dict = user_dict
        self.movie_dict = movie_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data.iloc[idx]["userId"]
        movie = self.data.iloc[idx]["movieId"]
        rating = self.data.iloc[idx]["rating"]

        user_idx = self.user_dict[user]
        movie_idx = self.movie_dict[movie]

        return user_idx, movie_idx, rating


# Define the model
class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super(MatrixFactorizationModel, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, user, movie):
        user_embedding = self.user_embedding(user)
        movie_embedding = self.movie_embedding(movie)

        x = torch.cat((user_embedding, movie_embedding), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x.view(-1)


# Create the dataset and data loaders
train_dataset = MovieLensDataset(train_data, user_dict, movie_dict)
test_dataset = MovieLensDataset(test_data, user_dict, movie_dict)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate the model and define the loss function and optimizer
model = MatrixFactorizationModel(num_users, num_movies)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for user, movie, rating in train_dataloader:
        optimizer.zero_grad()

        user = user.long()
        movie = movie.long()
        rating = rating.float()

        outputs = model(user, movie)
        loss = criterion(outputs, rating)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {running_loss / len(train_dataloader)}")

# Evaluate the model
model.eval()
test_loss = 0.0

with torch.no_grad():
    for user, movie, rating in test_dataloader:
        user = user.long()
        movie = movie.long()
        rating = rating.float()

        outputs = model(user, movie)
        loss = criterion(outputs, rating)

        test_loss += loss.item()

print(f"Test loss: {test_loss / len(test_dataloader)}")


# Recommend movies for a user
def recommend_movies(user_id, num_recommendations=5):
    user_idx = user_dict[user_id]

    user = torch.tensor([user_idx] * num_movies)
    movies = torch.tensor(list(movie_dict.values()))

    predictions = model(user, movies)
    _, top_movies = torch.topk(predictions, num_recommendations)

    top_movies = [k for k, v in movie_dict.items() if v in top_movies]
    recommended_movies = movies_data[movies_data["movieId"].isin(top_movies)]

    return recommended_movies[["title", "genres"]]


# Usage example
user_id = 42
recommendations = recommend_movies(user_id, num_recommendations=5)
print(f"Recommended movies for user {user_id}:")
print(recommendations)

# Save the trained model
torch.save(model.state_dict(), "movie_recommendation_model.pth")
