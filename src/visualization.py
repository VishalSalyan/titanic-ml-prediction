import matplotlib.pyplot as plt
import seaborn as sns

def plot_survival_distribution(df, save_path=None):
    sns.countplot(x='Survived', data=df)
    plt.title("Survival Distribution")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()


def plot_correlation_heatmap(df, save_path=None):
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()


def plot_feature_importance(model, feature_names, save_path=None):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=feature_names)
        plt.title("Feature Importance")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()