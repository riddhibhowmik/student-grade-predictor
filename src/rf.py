import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, ConfusionMatrixDisplay)

RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)

# plot styling
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

PALETTE = {
    'primary': '#2D5F8B',
    'accent': '#E07A5F',
    'muted': '#81B29A',
    'dark': '#3D405B',
}
CLASS_LABELS = ['A', 'B', 'C', 'D', 'F']

def data_processing(file_path):
    student_data = pd.read_csv(file_path)

    def convert_to_class(score):
        if score >= 90: return 0
        elif score >= 80: return 1
        elif score >= 70: return 2
        elif score >= 60: return 3
        else: return 4
        
    def convert_to_binary(value):
        return 1 if value == 'Yes' else 0
    
    features = ['study_hours_per_day', 'social_media_hours', 'part_time_job', 'attendance_percentage',
                'sleep_hours', 'exercise_frequency', 'mental_health_rating', 'extracurricular_participation']
    
    student_data = student_data.dropna(subset=features + ['exam_score'])
    student_data['target'] = student_data['exam_score'].apply(convert_to_class)
    student_data['part_time_job'] = student_data['part_time_job'].apply(convert_to_binary)
    student_data['extracurricular_participation'] = student_data['extracurricular_participation'].apply(convert_to_binary)

    # trees dont actually need normalization but we keep it consistent with the other files
    normalized_columns = ['study_hours_per_day', 'social_media_hours', 'attendance_percentage',
                            'sleep_hours', 'exercise_frequency', 'mental_health_rating']
    
    for column in normalized_columns:
        student_data[column] = (student_data[column] - student_data[column].min()) / (student_data[column].max() - student_data[column].min())

    student_data = student_data.drop(columns=['exam_score'])
    student_data = student_data[features + ['target']]

    return student_data

student_data = data_processing('data/student_habits_performance.csv')
X = student_data.drop(columns=['target'])
y = student_data['target']

print("class distribution:")
print(y.value_counts().sort_index().rename(lambda i: CLASS_LABELS[i]))
print()

# 70/15/15 stratified split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

# hyperparameter tuning
print("tuning hyperparameters with randomized search (this takes a minute)...")

param_distributions = {
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.8],
    'bootstrap': [True],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

# fix n_estimators at 400 during tuning (enough for stable CV); the actual
# final n_estimators is picked below via OOB early stopping
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
base_rf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='accuracy',
    cv=cv_strategy,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

random_search.fit(X_train, y_train)

print(f"\nbest cv accuracy: {random_search.best_score_ * 100:.2f}%")
print("best hyperparameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

tuned_params = dict(random_search.best_params_)

# OOB-based "early stopping" for n_estimators. RF trees train independently so we cant
# stop mid-fit like a neural net, but we can grow the forest incrementally with warm_start
# and watch the OOB score. once it plateaus for `patience` steps in a row, stop adding trees.
print("\ngrowing forest incrementally, monitoring OOB score...")

step_size = 25
max_trees = 1000
patience = 5

rf_grow = RandomForestClassifier(
    n_estimators=0, warm_start=True, oob_score=True,
    random_state=RANDOM_STATE, n_jobs=-1, **tuned_params
)

oob_history = []
best_oob = -np.inf
best_n_estimators = step_size
counter = 0

for n in range(step_size, max_trees + 1, step_size):
    rf_grow.n_estimators = n
    rf_grow.fit(X_train, y_train)
    oob = rf_grow.oob_score_
    oob_history.append((n, oob))

    improved = oob > best_oob + 1e-4
    if improved:
        best_oob = oob
        best_n_estimators = n
        counter = 0
    else:
        counter += 1

    if n % 100 == 0 or improved:
        tag = " <- new best" if improved else ""
        print(f"  trees: {n}, OOB: {oob:.4f}{tag}")

    if counter >= patience:
        print(f"Early stopping triggered at n_estimators={n} (no OOB improvement for {patience} steps)")
        break
else:
    print(f"Reached max_trees={max_trees} without early stopping")

print(f"chosen n_estimators: {best_n_estimators} (OOB: {best_oob*100:.2f}%)")

# final model with the OOB-chosen n_estimators
best_rf = RandomForestClassifier(
    n_estimators=best_n_estimators, oob_score=True,
    random_state=RANDOM_STATE, n_jobs=-1, **tuned_params
)
best_rf.fit(X_train, y_train)
print(f"final OOB score: {best_rf.oob_score_ * 100:.2f}%")

val_predictions = best_rf.predict(X_val)
print(f"validation accuracy: {accuracy_score(y_val, val_predictions) * 100:.2f}%")

print("\nevaluating on test set...")
predictions = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Random Forest Real World accuracy: {accuracy * 100:.2f}%")
print("\nclassification report:")
print(classification_report(y_test, predictions, target_names=CLASS_LABELS, zero_division=0))

# ===== visualizations =====

# FIGURE 1: confusion matrix (raw + row-normalized)
cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3, 4])
cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

disp1 = ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS)
disp1.plot(ax=axes[0], cmap='Blues', colorbar=False, values_format='d')
axes[0].set_title('Raw Counts')
axes[0].grid(False)

disp2 = ConfusionMatrixDisplay(cm_normalized, display_labels=CLASS_LABELS)
disp2.plot(ax=axes[1], cmap='Blues', colorbar=False, values_format='.2f')
axes[1].set_title('Row-Normalized (per-class recall)')
axes[1].grid(False)

fig.suptitle('Random Forest — Confusion Matrices', fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('rf_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# FIGURE 2: per-class precision / recall / F1
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, predictions, labels=[0, 1, 2, 3, 4], zero_division=0
)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(CLASS_LABELS))
width = 0.26

bars1 = ax.bar(x - width, precision, width, label='Precision', color=PALETTE['primary'], alpha=0.9)
bars2 = ax.bar(x, recall, width, label='Recall', color=PALETTE['accent'], alpha=0.9)
bars3 = ax.bar(x + width, f1, width, label='F1', color=PALETTE['muted'], alpha=0.9)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=9)

xtick_labels = [f'{CLASS_LABELS[i]}\n(n={support[i]})' for i in range(len(CLASS_LABELS))]
ax.set_xticks(x)
ax.set_xticklabels(xtick_labels)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.15)
ax.set_title('Per-Class Performance (test set)')
ax.axhline(accuracy, color=PALETTE['dark'], linestyle=':', alpha=0.5)
ax.legend(loc='upper right', frameon=True, framealpha=0.95)
ax.text(len(CLASS_LABELS) - 0.5, accuracy + 0.015,
        f'overall acc: {accuracy:.2f}', fontsize=9, color=PALETTE['dark'],
        ha='right', style='italic')

plt.tight_layout()
plt.savefig('rf_per_class_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# FIGURE 3: OOB trajectory with early-stopping point marked
oob_n = [n for n, _ in oob_history]
oob_vals = [s for _, s in oob_history]

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(oob_n, oob_vals, marker='o', linewidth=2.2, markersize=6,
        color=PALETTE['primary'], label='OOB accuracy')

# shade the patience window where OOB stopped improving
stop_n = oob_n[-1]
if stop_n > best_n_estimators:
    ax.axvspan(best_n_estimators, stop_n, alpha=0.12,
               color=PALETTE['accent'], label=f'patience window ({patience} steps)')

ax.axvline(best_n_estimators, color=PALETTE['muted'], linestyle='--', alpha=0.8,
           label=f'chosen n_estimators = {best_n_estimators}')
ax.axhline(best_oob, color=PALETTE['dark'], linestyle=':', alpha=0.4)

ax.set_xlabel('Number of Trees (n_estimators)')
ax.set_ylabel('OOB Accuracy')
ax.set_title('Incremental Forest Growth with OOB Early Stopping')
ax.legend(loc='lower right', frameon=True, framealpha=0.95)

plt.tight_layout()
plt.savefig('rf_learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nall figures saved as PNG files in current directory.")