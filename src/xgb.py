import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def data_processing(file_path):
    student_data = pd.read_csv(file_path)

    def convert_to_class(score):
        if score >= 90: # A
            return 0
        elif score >= 80: # B
            return 1
        elif score >= 70: # C
            return 2
        elif score >= 60: # D
            return 3
        else: # F
            return 4
        
    def convert_to_binary(value):
        if value == 'Yes':
            return 1
        else:
            return 0
    
    features = ['study_hours_per_day', 'social_media_hours', 'part_time_job', 'attendance_percentage',
                'sleep_hours', 'exercise_frequency', 'mental_health_rating', 'extracurricular_participation']
    
    student_data = student_data.dropna(subset=features + ['exam_score'])
    student_data['target'] = student_data['exam_score'].apply(convert_to_class)
    student_data['part_time_job'] = student_data['part_time_job'].apply(convert_to_binary)
    student_data['extracurricular_participation'] = student_data['extracurricular_participation'].apply(convert_to_binary)

    normalized_columns = ['study_hours_per_day', 'social_media_hours', 'attendance_percentage',
                            'sleep_hours', 'exercise_frequency', 'mental_health_rating']
    
    for column in normalized_columns:
        student_data[column] = (student_data[column] - student_data[column].min()) / (student_data[column].max() - student_data[column].min())

    student_data = student_data.drop(columns=['exam_score'])
    student_data = student_data[features + ['target']]

    return student_data

# load data
student_data = data_processing('data/student_habits_performance.csv')
X = student_data.drop(columns=['target'])
y = student_data['target']

# split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

# initialize xgboost
print ("training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    eval_metric='mlogloss',
    early_stopping_rounds=10,
    random_state=1,
    # make sure it doesnt overthink since data isnt super big
    max_depth=3,
    # slow down learning so it can find broader patterns, make sure no overfitting
    learning_rate=0.05,
    # more trees, since the learning rate is slower, will help it find more complex patterns,
    # but still avoid overfitting
    n_estimators=500,
    # look at 80% of students at a time to really make sure no overfitting happens
    subsample=0.8
)

# train xhboost model
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# evaluate w test set
predictions = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"XGBoost Real World accuracy: {accuracy * 100:.2f}%")

# confusion matrix generation
print("generating xgboost confusion matrix...")
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=[0, 1, 2, 3, 4],
    display_labels=['A', 'B', 'C', 'D', 'F'],
    cmap='Greens'
)
plt.show()
