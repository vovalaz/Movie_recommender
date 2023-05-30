import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


movies_data = pd.read_csv("movies.csv")
ratings_data = pd.read_csv("ratings.csv")


class RecommendationSystem(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(RecommendationSystem, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, movie_ids):
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        concatenated = torch.cat((user_embedded, movie_embedded), dim=0)
        output = self.fc(concatenated)
        return output.squeeze()


def train_model(model, data, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for user_id, movie_id, rating in data:
            user_id = torch.tensor(user_id, dtype=torch.long)
            movie_id = torch.tensor(movie_id, dtype=torch.long)
            rating = torch.tensor(rating, dtype=torch.float)

            optimizer.zero_grad()
            predicted_rating = model(user_id, movie_id)
            loss = criterion(predicted_rating, rating)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def generate_movie_recommendations(model, user_id, top_k=5):
    user_id = torch.tensor(user_id, dtype=torch.long)

    user_embedding = model.user_embedding(user_id)

    all_movie_ids = torch.arange(model.movie_embedding.num_embeddings, dtype=torch.long)
    movie_embeddings = model.movie_embedding(all_movie_ids)

    similarity_scores = torch.cosine_similarity(user_embedding, movie_embeddings, dim=1)

    top_indices = torch.topk(similarity_scores, k=top_k).indices

    movie_recommendations = [
        movies_data.loc[movies_data["movie_id"] == idx.item(), "title"].values[0] for idx in top_indices
    ]
    return movie_recommendations


def continuous_train(new_data, num_epochs, learning_rate):
    model = RecommendationSystem(num_users, num_movies, embedding_dim)
    model.load_state_dict(torch.load("content_collab_model.pt"))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for user_id, movie_id, rating in new_data:
            user_id = torch.tensor(user_id, dtype=torch.long)
            movie_id = torch.tensor(movie_id, dtype=torch.long)
            rating = torch.tensor(rating, dtype=torch.float)

            optimizer.zero_grad()
            predicted_rating = model(user_id, movie_id)
            loss = criterion(predicted_rating, rating)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(new_data)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "content_collab_model.pt")


num_users = max(ratings_data["user_id"]) + 1
num_movies = max(movies_data["movie_id"]) + 1
embedding_dim = 32
num_epochs = 10
learning_rate = 0.001

if __name__ == "__main__":
    # train_data = [tuple(row) for _, row in ratings_data.iterrows()]

    # model = RecommendationSystem(num_users, num_movies, embedding_dim)
    # train_model(model, train_data, num_epochs, learning_rate)
    # torch.save(model.state_dict(), "content_collab_model.pt")
    model = RecommendationSystem(num_users, num_movies, embedding_dim)
    model.load_state_dict(torch.load("content_collab_model.pt"))

    # Generate movie recommendations for user with user_id = 1
    user_id = 25
    recommendations = generate_movie_recommendations(model, user_id, top_k=5)

    print(f"Movie Recommendations for User {user_id}:")
    for i, movie in enumerate(recommendations):
        print(f"{i+1}. {movie}")
