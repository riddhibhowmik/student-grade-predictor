import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    
    data['grade'] = data['exam_score'].apply(convert_to_grade)
    
    # preprocess data 
    columns = ['study_hours_per_day', 'social_media_hours', 'part_time_job', 'attendance_percentage',
               'sleep_hours', 'exercise_frequency', 'mental_health_rating', 'extracurricular_participation']

    data = data.dropna(subset=columns + ['grade'])

    # TODO: turn the part_time_job and extracurricular_participation into binary, for now just
    # use the columns that are numbers already
    numeric_data = ['study_hours_per_day', 'social_media_hours', 'attendance_percentage',
                    'sleep_hours', 'exercise_frequency', 'mental_health_rating']
    x = data[numeric_data]
    y = data['grade']

    # split data into training/testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # train a baseline model (can train more complex models later, starting with simple logistic regression)
    # XGBoost, Random Forest, etc
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # results
    acc = accuracy_score(y_test, model.predict(x_test))
    print(f"baseline accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()