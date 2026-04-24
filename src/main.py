import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    # load dataset
    try: 
        data = pd.read_csv('data/student_habits_performance.csv')
    except FileNotFoundError:
        print("csv not found")
        return
    
    def convert_to_grade(score):
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
        
    def convert_to_binary(value):
        if value == 'Yes':
            return 1
        else:
            return 0
    
    data['grade'] = data['exam_score'].apply(convert_to_grade)
    data['part_time_job'] = data['part_time_job'].apply(convert_to_binary)
    data['extracurricular_participation'] = data['extracurricular_participation'].apply(convert_to_binary)
    
    # preprocess data 
    columns = ['study_hours_per_day', 'social_media_hours', 'part_time_job', 'attendance_percentage',
                'sleep_hours', 'exercise_frequency', 'mental_health_rating', 'extracurricular_participation']

    data = data.dropna(subset=columns + ['grade'])

    x = data[columns]
    y = data['grade']

    # standardize features 
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # split data into training and temporary sets (70% train, 30% temp)
    x_train, x_temp, y_train, y_temp = train_test_split(x_scaled, y, test_size=0.3, random_state=1)

    # split remainder in half into validation and test sets
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)
    
    # train a baseline model (can train more complex models later, starting with simple logistic regression)
    # XGBoost, Random Forest, etc
    model = LogisticRegression(max_iter=5000)
    model.fit(x_train, y_train)

    # results
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)

    print(f"baseline accuracy: {acc:.2%}")

    print("confusion matrix:")
    ConfusionMatrixDisplay.from_predictions (
        y_test,
        predictions, 
        labels = ['A', 'B', 'C', 'D', 'F'],
        cmap = 'Blues'
    )
    plt.show()

    # visualize feature importance (for logistic reg, itll be the coefficients)
    print("visualizing feature importance...")

    # average the absolite value of coefficients across all 5 classes to get overall score
    importance = np.mean(np.abs(model.coef_), axis=0)
    sort_indices = np.argsort(importance)
    sort_names = [columns[i] for i in sort_indices]

    # plot bar chart
    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(sort_names))

    # add the bars and labels to plot
    ax.barh(y_pos, importance[sort_indices], color='#2D5F8B', alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sort_names)
    ax.set_xlabel('feature importance score')
    ax.set_title('Logistic Regression Baseline: How much does each feature contribute to the prediction?')

    # add importance values next to the bars
    for i, v in enumerate(importance[sort_indices]):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, color='#3D405B')

    # make it look borderless like on other models
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.25, linestyle='--')

    # compress if needed so it fits and show it
    plt.tight_layout()
    plt.savefig('lr_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()