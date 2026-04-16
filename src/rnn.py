import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class RNN(nn.Module):
    def __init__(self, input_size, hidden_layer):
        super(RNN, self).__init__()
        
        self.hidden_layer = hidden_layer
        self.numOfLayer = 2
        
        self.rnn = nn.RNN(input_size, self.hidden_layer, self.numOfLayer, nonlinearity='tanh', batch_first=True)
        
        self.W = nn.Linear(self.hidden_layer, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.loss = nn.NLLLoss()
        
    def compute_Loss(self, predicted_vector, target_label):
        return self.loss(predicted_vector, target_label)
    
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        
        out, hidden = self.rnn(inputs)
        
        final_output = out[:, -1, :]
        
        output_layer_reps = self.W(final_output)
        
        predicted_vector = self.softmax(output_layer_reps)
        
        return predicted_vector

def data_processing(file_path):
    student_data = pd.read_csv(file_path)
    
    def convert_to_pass_fail(score):
        if score >= 70:
            return 1
        else:
            return 0
        
    def convert_to_binary(value):
        if value == 'Yes':
            return 1
        else:
            return 0
    
    features = ['study_hours_per_day', 'social_media_hours', 'part_time_job', 'attendance_percentage',
                'sleep_hours', 'exercise_frequency', 'mental_health_rating', 'extracurricular_participation']
    
    student_data = student_data.dropna(subset=features + ['exam_score'])
    
    student_data['target'] = student_data['exam_score'].apply(convert_to_pass_fail)
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
y_train_tensor = torch.tensor(target_train.values, dtype=torch.long)

X_val_tensor = torch.tensor(features_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(target_val.values, dtype=torch.long)

X_test_tensor = torch.tensor(features_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(target_test.values, dtype=torch.long)


model = RNN(input_size=X_train_tensor.shape[1], hidden_layer=16)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def calculate_accuracy(predictions, targets):
    rounded_predictions = torch.argmax(predictions, dim=1)
    correct = (rounded_predictions == targets).sum().item()
    
    accuracy = correct / targets.shape[0]
    
    return accuracy

print("Starting training...")
num_epochs = 100

best_val_loss = float('inf')
patience = 10
counter = 0

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
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered at epoch {}".format(epoch + 1))
        break

print("Training complete.")

print("\nEvaluating on test set...")

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    test_accuracy = calculate_accuracy(test_predictions, y_test_tensor)
    
    print(f'Real World Accuracy: {test_accuracy*100:.2f}%')