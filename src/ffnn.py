import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# set the random seed so its the same for ffnn and rnn,  betetr comparison
torch.manual_seed(1)

class FFNN(nn.Module):
    def __init__(self, input_size):
        super(FFNN, self).__init__()
        
        self.hidden_layer_1 = nn.Linear(input_size, 16)
        self.relu1 = nn.ReLU()
        
        self.hidden_layer_2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        # change output to 5 for 5 classes (A, B, C, D, F)
        self.hidden_layer_3 = nn.Linear(8, 5)
        # we don't need sigmoid, cross entropy loss will handle probabilities
        #self.output_layer = nn.Sigmoid()

    def forward(self, x):
        out = self.hidden_layer_1(x)
        out = self.relu1(out)
        
        out = self.hidden_layer_2(out)
        out = self.relu2(out)
        
        out = self.hidden_layer_3(out)
        # got rid of sigmoid, so don't need this
        #out = self.output_layer(out)
        
        return out

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

student_data = data_processing('data/student_habits_performance.csv')

features_df = student_data.drop(columns=['target'])
target_df = student_data['target']

# split data into training and temporary sets (70% train, 30% temp)
features_train, features_temp, target_train, target_temp = train_test_split(features_df, target_df, test_size=0.3, random_state=1)

# split remainder in half into validation and test sets
features_val, features_test, target_val, target_test = train_test_split(features_temp, target_temp, test_size=0.5, random_state=1)

# convert to tensors
X_train_tensor = torch.tensor(features_train.values, dtype=torch.float32)
# change to long for multi-class classification 
y_train_tensor = torch.tensor(target_train.values, dtype=torch.long)

X_val_tensor = torch.tensor(features_val.values, dtype=torch.float32)
# change to long for multi-class classification
y_val_tensor = torch.tensor(target_val.values, dtype=torch.long)

X_test_tensor = torch.tensor(features_test.values, dtype=torch.float32)
# change to long for multi-class classification
y_test_tensor = torch.tensor(target_test.values, dtype=torch.long)


model = FFNN(input_size=X_train_tensor.shape[1])
# use cross entropy instead of BCE, since multiple classes and not binary
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def calculate_accuracy(predictions, targets):
    # use argmax to get predicted class from the output probabilities, set 
    # dim = 1 to get index of max value
    rounded_predictions = torch.argmax(predictions, dim=1)
    correct = (rounded_predictions == targets).sum().item()
    
    accuracy = correct / targets.shape[0]
    
    return accuracy

print("Starting training...")
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    
    predicted = model(X_train_tensor)
    
    loss = criterion(predicted, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = criterion(val_predictions, y_val_tensor)
        val_accuracy = calculate_accuracy(val_predictions, y_val_tensor)
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.1f}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy*100:.2f}%')

print("Training complete.")

print("\nEvaluating on test set...")

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    test_accuracy = calculate_accuracy(test_predictions, y_test_tensor)
    
    print(f'Real World Accuracy: {test_accuracy*100:.2f}%')

# confusion matrix for visualization
print("generating ffnn confusion matrix...")

# turn the pytorch tensors back into numpy arrays so it works w/ confusion matrix
predicted_classes = torch.argmax(test_predictions, dim=1).numpy()
true_classes = y_test_tensor.numpy()

ConfusionMatrixDisplay.from_predictions (
    true_classes,
    predicted_classes,
    labels = [0, 1, 2, 3, 4],
    display_labels=['A', 'B', 'C', 'D', 'F'],
    cmap = 'Purples'
)
plt.show()