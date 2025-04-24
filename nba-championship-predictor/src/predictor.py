import pandas as pd
from models.model import ChampionshipPredictor
from utils.data_processing import load_data, preprocess_data, split_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class NBAChampionshipPredictor:
    def __init__(self):
        self.model = ChampionshipPredictor()
        self.data = None

    def load_and_prepare_data(self, file_path, train_seasons=None, predict_season=2025):
        self.data = load_data(file_path)
        self.data = preprocess_data(self.data)
        

        # Filter out data from before 1999
        self.data = self.data[self.data['season'] >= 1999]
        print(f"Total champions in the dataset: {self.data[self.data['champion'] == 1].shape[0]}")
        print(f"Data includes seasons from {self.data['season'].min()} to {self.data['season'].max()}")

        # Split the data into training and prediction sets
        if train_seasons:
            self.train_data = self.data[self.data['season'].isin(train_seasons)]
        else:
            self.train_data = self.data[self.data['season'] != predict_season]

        self.predict_data = self.data[self.data['season'] == predict_season]

        # Fit the scaler on the training data
        numerical_features = ['o_rtg', 'd_rtg', 'win_percentage', 'mov', 'pace', 'ts_percent']
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_data[numerical_features])

    def train(self):
        X_train, X_test, y_train, y_test = split_data(self.data)
        self.model.train_model(X_train, y_train)

        print("Training set class distribution:")
        print(y_train.value_counts())

        print("\nTest set class distribution:")
        print(y_test.value_counts())
        
        # Get predictions for the test set
        y_pred = self.model.predict(X_test)  # Class predictions (0 or 1)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
        
        # Calculate metrics
        accuracy = self.model.evaluate_model(X_test, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Print metrics
        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}  % of predicted champions that were correct")
        print(f"Recall:    {recall:.4f}    % of actual champions correctly predicted")
        print(f"F1-Score:  {f1:.4f}")         # Harmonic mean of precision/recall
        
        # Detailed report
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=['Non-Champion', 'Champion']))
        
        # Confusion matrix (optional)
        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))

    def predict(self, input_data):
        # Convert input_data to a DataFrame with the same feature names
        input_df = pd.DataFrame(input_data, columns=['o_rtg', 'd_rtg', 'win_percentage'])

        # Normalize the input data using the stored scaler
        numerical_features = ['o_rtg', 'd_rtg', 'win_percentage']
        input_df[numerical_features] = self.scaler.transform(input_df[numerical_features])  # Normalize input data

        # Predict using the trained model
        return self.model.predict_championship(input_df)

    def predict_champion(self):
        """Predict the champion with original (unscaled) values and normalize probabilities."""
        numerical_features = ['o_rtg', 'd_rtg', 'win_percentage', 'mov', 'pace', 'ts_percent', 'f_tr', 'x3p_ar', 'age', 'srs']
        X = self.predict_data[numerical_features]

        # Ensure predict_data is a proper copy
        self.predict_data = self.predict_data.copy()

        # Store raw probabilities using .loc to avoid SettingWithCopyWarning
        self.predict_data.loc[:, 'champion_probability'] = self.model.predict_proba(X)[:, 1]

        # Normalize probabilities so they sum to 1
        total_probability = self.predict_data['champion_probability'].sum()
        self.predict_data.loc[:, 'champion_probability'] /= total_probability

        # Get the predicted champion (team with the highest normalized probability)
        champion_idx = self.predict_data['champion_probability'].idxmax()
        predicted_champion = self.predict_data.loc[champion_idx].copy()

        # Return with original values
        return pd.Series({
            'team': predicted_champion['team'],
            'srs': predicted_champion['srs_original'],
            'o_rtg': predicted_champion['o_rtg_original'],
            'd_rtg': predicted_champion['d_rtg_original'],
            'win_percentage': predicted_champion['win_percentage_original'],
            'mov': predicted_champion['mov_original'],
            'pace': predicted_champion['pace_original'],
            'ts_percent': predicted_champion['ts_percent_original'],
            'f_tr': predicted_champion['f_tr_original'],
            'x3p_ar': predicted_champion['x3p_ar_original'],
            'age': predicted_champion['age_original'],
            'champion_probability': predicted_champion['champion_probability']
        })
    
    def visualize_top_teams(self):
        """Visualize the top 5 teams with the highest champion probabilities."""
        # Sort the prediction data by champion probability in descending order
        top_teams = self.predict_data.sort_values(by='champion_probability', ascending=False).head(8)

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(top_teams['team'], top_teams['champion_probability'], color='yellow')
        plt.title('Top 8 Teams with Highest Champion Probabilities (2025)', fontsize=16)
        plt.xlabel('Team', fontsize=14)
        plt.ylabel('Champion Probability', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.tight_layout()
        plt.show()

    def visualize_stat_relationship(self, stat, title):
        """Visualize the relationship between a stat and winning a championship using box plots."""
        plt.figure(figsize=(10, 6))

        # Create a box plot
        self.data.boxplot(column=stat, by='champion', grid=False, patch_artist=True, showmeans=True,
                          boxprops=dict(facecolor='yellow', color='blue'),
                          medianprops=dict(color='red'),
                          meanprops=dict(marker='o', markerfacecolor='yellow', markersize=8),
                          whiskerprops=dict(color='black'),
                          capprops=dict(color='black'))

        # Add labels and title
        plt.title(title, fontsize=16)
        plt.suptitle('')  # Remove the default "Boxplot grouped by champion" title
        plt.xlabel('Championship Status (0 = Non-Champion, 1 = Champion)', fontsize=14)
        plt.ylabel(stat, fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_feature_importance(self, feature_importance, feature_names):
        """Visualize the feature importance as a bar chart."""
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importance, color='yellow')
        plt.xlabel('Feature Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.title('Feature Importance for Championship Prediction', fontsize=16)
        plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
        plt.tight_layout()
        plt.show()

    def visualize_correlation_heatmap(self):
        """Generate and display a heatmap of feature correlations with descriptive labels."""
        # Rename columns for better interpretability
        column_mapping = {
            'o_rtg_original': 'Offensive Rating',
            'd_rtg_original': 'Defensive Rating',
            'win_percentage_original': 'Win Percentage',
            'mov_original': 'Margin of Victory',
            'pace_original': 'Pace of Play',
            'ts_percent_original': 'True Shooting Percentage',
            'f_tr_original': 'Free Throw Rate',
            'x3p_ar': '3-Point Attempt Rate',
            'age_original': 'Average Age',
            'srs_original': 'Simple Rating System',
            'champion': 'Champion'
        }

        # Select relevant columns and rename them
        corr_matrix = self.data[list(column_mapping.keys())].rename(columns=column_mapping).corr()

        # Print the correlation of features with the 'Champion' column
        champion_corr = corr_matrix['Champion'].sort_values(ascending=False)
        print("\n=== Feature Correlation Analysis ===")
        print(champion_corr.to_markdown())  # Pretty-print table

        # Generate the heatmap
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
        plt.title("Feature Correlation Matrix", fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')  # Save the heatmap as an image
        plt.show()

    # Feature names corresponding to the importance values
    feature_names = [
        "Offensive Rating",
        "Defensive Rating",
        "Win Percentage",
        "Pace of Play",
        "Margin of Victory",
        "True Shooting Percentage",
        "Free Throw Rate",
        "3-Point Attempt Rate",
        "Simple Rating System",
        "Average Age"
    ]

if __name__ == "__main__":
    predictor = NBAChampionshipPredictor()
    predictor.load_and_prepare_data('../data/Team_Summaries2.csv', predict_season=2025)  # Load only the 2025 season
    predictor.train()  # Train the model

    # Visualize the correlation heatmap
    predictor.visualize_correlation_heatmap()

    # Retrieve feature importance values from the trained model
    feature_importance = predictor.model.get_feature_importance()
    
    # Print a header to explain what the feature importance values represent
    print("\n===Feature Importance===")    
    print(f"o_rtg (Offensive Rating): {feature_importance[0]:.4f}")
    print(f"d_rtg (Defensive Rating): {feature_importance[1]:.4f}")
    print(f"win_percentage (Win Percentage): {feature_importance[2]:.4f}")
    print(f"pace (Pace of Play): {feature_importance[3]:.4f}")
    print(f"mov (Margin of Victory): {feature_importance[4]:.4f}")
    print(f"ts_percent (True Shooting Percentage): {feature_importance[5]:.4f}")
    print(f"f_tr (Free Throw Rate): {feature_importance[6]:.4f}")
    print(f"x3p_ar (3-Point Attempt Rate): {feature_importance[7]:.4f}")
    print(f"srs (Simple Rating System): {feature_importance[8]:.4f}")
    print(f"age (Average Age): {feature_importance[9]:.4f}")

    # Predict the 2025 NBA Champion
    predicted_champion = predictor.predict_champion()
    print("\n===Predicted 2025 NBA Champion:===")
    print(predicted_champion[['team', 'o_rtg', 'd_rtg', 'win_percentage', 'champion_probability']])

    # Visualize feature importance
    predictor.visualize_feature_importance(feature_importance, predictor.feature_names)
    # Visualize the top 5 teams
    predictor.visualize_top_teams()