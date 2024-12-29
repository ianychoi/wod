import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

pd.set_option('future.no_silent_downcasting', True)

# Load the CSV file
data = pd.read_csv('exercise_analysis-from-741-posts.csv')

# Convert columns to binary
def convert_to_binary(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0 if str(x).upper() == 'FALSE' else np.nan)
    return df

# Plot the PR curve
def plot_pr_curve(y_true, precisions, recalls):
    # Draw the PR curve
    plt.figure(figsize=(8,6))
    plt.plot(recalls, precisions, 'b-', label='PR curve')
    
    # Decorate the plot
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    
    plt.show()

# Calculate the values for the PR curve
def calculate_for_auprc(TP, TN, FP, FN):
    TP = TP.astype(int)
    TN = TN.astype(int)
    FP = FP.astype(int)
    FN = FN.astype(int)

    y_true = [1] * len(TP) + [0] * len(FP)
    y_scores = list(TP) + list(FP)
    return y_true, y_scores

# Calculate classification metrics
def calculate_classification_metrics(predictions, actuals):
    # Remove rows with missing values
    valid_mask = ~(predictions.isna() | actuals.isna())
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]
    
    TP_array = ((predictions == 1) & (actuals == 1))
    TN_array = ((predictions == 0) & (actuals == 0))
    FP_array = ((predictions == 1) & (actuals == 0))
    FN_array = ((predictions == 0) & (actuals == 1))

    TP = TP_array.sum()
    TN = TN_array.sum()
    FP = FP_array.sum()
    FN = FN_array.sum()
    
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    y_true, y_scores = calculate_for_auprc(TP_array, TN_array, FP_array, FN_array)

    return accuracy, precision, recall, total, TP, TN, FP, FN, y_true, y_scores

# Calculate time metrics
def calculate_time_metrics(data, model_time_column):
    # Select only the data where duration_estimation is FALSE
    subset = data[ (data['duration_estimation'] != True) ]

    subset['human_time'] = subset['human_time'].fillna(0)

    # Select only the data where model_time_column is not NA
    valid_data = subset[['human_is_exercise', 'human_time', model_time_column]]

    # Calculate the predictions (True if the difference is within 1)
    predictions = abs(valid_data['human_time'] - valid_data[model_time_column]) <= 1
    
    # True Positive: Actual exercise and time predicted correctly
    TP_array = ((valid_data['human_is_exercise'] == True) & predictions )
    
    # True Negative: Actual not exercise and time not predicted correctly
    TN_array = ((valid_data['human_is_exercise'] == False) & ~predictions )
    
    # False Positive: Actual not exercise but time predicted correctly
    FP_array = ((valid_data['human_is_exercise'] == False) & predictions )
    
    # False Negative: Actual exercise but time not predicted correctly
    FN_array = ((valid_data['human_is_exercise'] == True) & ~predictions )

    TP = TP_array.sum()
    TN = TN_array.sum()
    FP = FP_array.sum()
    FN = FN_array.sum()

    total = len(valid_data)
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    y_true, y_scores = calculate_for_auprc(TP_array, TN_array, FP_array, FN_array)

    return {
        'count': total,
        'tp': TP,
        'tn': TN,
        'fp': FP,
        'fn': FN,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'y_true': y_true,
        'y_scores': y_scores
    }

def calculate_calorie_metrics(data, model_time_column):
    # Select only the data where calorie_estimation is FALSE
    subset = data[ (data['calorie_estimation'] != True) ]

    subset['human_time'] = subset['human_time'].fillna(0)

    # Select only the data where model_time_column is not NA
    valid_data = subset[['human_is_exercise', 'human_calories', model_time_column]]

    # Calculate the predictions (True if the difference is within 1)
    predictions = abs(valid_data['human_calories'] - valid_data[model_time_column]) <= 1
    
    # True Positive: Actual exercise and time predicted correctly
    TP_array = ((valid_data['human_is_exercise'] == True) & predictions )
    
    # True Netative: Actual not exercise and time not predicted correctly
    TN_array = ((valid_data['human_is_exercise'] == False) & ~predictions )
    
    # False Positive: Actual not exercise but time predicted correctly
    FP_array = ((valid_data['human_is_exercise'] == False) & predictions )
    
    # False Negative: Actual exercise but time not predicted correctly
    FN_array = ((valid_data['human_is_exercise'] == True) & ~predictions )

    TP = TP_array.sum()
    TN = TN_array.sum()
    FP = FP_array.sum()
    FN = FN_array.sum()

    total = len(valid_data)
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    y_true, y_scores = calculate_for_auprc(TP_array, TN_array, FP_array, FN_array)
    
    return {
        'count': total,
        'tp': TP,
        'tn': TN,
        'fp': FP,
        'fn': FN,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'y_true': y_true,
        'y_scores': y_scores
    }

# Calculate confidence interval
def calculate_confidence_interval(successes, total, confidence_level=0.95):
    """
    Calculate confidence interval for a proportion
    """
    if total == 0:
        return 0, 0
        
    prop = successes / total
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    se = np.sqrt((prop * (1 - prop)) / total)
    ci_low = max(0, prop - z * se)
    ci_upp = min(1, prop + z * se)
    
    return ci_low, ci_upp

# Print metrics table
def print_metrics_table(title, results_dict):
    print(f"\n=== {title} ===")
    
    rows = []
    for model_name in results_dict.keys():
        result = results_dict[model_name]
        
        # Calculate confidence intervals
        acc_low, acc_upp = calculate_confidence_interval(
            result['tp'] + result['tn'], 
            result['count']
        )
        prec_low, prec_upp = calculate_confidence_interval(
            result['tp'], 
            result['tp'] + result['fp']
        )
        rec_low, rec_upp = calculate_confidence_interval(
            result['tp'], 
            result['tp'] + result['fn']
        )
        
        rows.append({
            'Model': model_name,
            'Count': result['count'],
            'TP': result['tp'],
            'TN': result['tn'],
            'FP': result['fp'],
            'FN': result['fn'],
            # 'Accuracy': f"{result['accuracy']:.3%} ",
            # 'Precision': f"{result['precision']:.3%} ",
            # 'Recall': f"{result['recall']:.3%} "
            'Accuracy': f"{result['accuracy']:.3%} ({acc_low:.3%}-{acc_upp:.3%})",
            'Precision': f"{result['precision']:.3%} ({prec_low:.3%}-{prec_upp:.3%})",
            'Recall': f"{result['recall']:.3%} ({rec_low:.3%}-{rec_upp:.3%})"
        })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


# Data pre-processing
exercise_columns = ['human_is_exercise', 'openai_is_exercise', 
                   'aoai_is_exercise', 'bedrock_is_exercise']
data = convert_to_binary(data, exercise_columns)
data = convert_to_binary(data, ['duration_estimation', 'calorie_estimation'])

# Dictionary to store results
exercise_results = {}
time_results = {}
calorie_results = {}

# 1. Exercise classification performance evaluation
models = {
    'OpenAI': 'openai_is_exercise',
    'Azure OpenAI': 'aoai_is_exercise',
    'Bedrock': 'bedrock_is_exercise'
}

for model_name, column in models.items():
    acc, prec, rec, total, tp, tn, fp, fn, y_true, y_scores = calculate_classification_metrics(
        data[column], data['human_is_exercise']
    )
    exercise_results[model_name] = {
        'count': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'y_true': y_true,
        'y_scores': y_scores
    }

# 2. Duration prediction performance evaluation
time_models = {
    'OpenAI': 'openai_exercise_time',
    'Azure OpenAI': 'aoai_exercise_time',
    'Bedrock': 'bedrock_exercise_time'
}

for model_name, time_column in time_models.items():
    time_results[model_name] = calculate_time_metrics(data, time_column)

# 3. Calorie prediction performance evaluation
calorie_models = {
    'OpenAI': 'openai_exercise_calories',
    'Azure OpenAI': 'aoai_exercise_calories',
    'Bedrock': 'bedrock_exercise_calories'
}

for model_name, calorie_column in calorie_models.items():
    calorie_results[model_name] = calculate_calorie_metrics(data, calorie_column)


# 결과 출력
print_metrics_table("Is_Exercise Classification - Performance Evaluation", exercise_results)

print_metrics_table("Exercise Time Prediction - Performance Evaluation", time_results)

print_metrics_table("Exercise Calorie Prediction - Performance Evaluation", calorie_results)

def plot_pr_curves(y_true, predictions_dict, title, row_index):
    # Set consistent colors for each model
    model_colors = {
        'OpenAI': '#1f77b4',        # Default matplotlib blue
        'Azure OpenAI': '#ff7f0e',  # Default matplotlib orange
        'Bedrock': '#2ca02c'        # Default matplotlib green
    }

    col_index = 1
    for model_name, y_pred in predictions_dict.items():
        subplot_pos = (row_index-1) * 3 + col_index
        plt.subplot(3, 3, subplot_pos)
        precision, recall, _ = precision_recall_curve(y_true[model_name], y_pred)
        auprc = average_precision_score(y_true[model_name], y_pred)
        #auprc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (AUPRC = {auprc:.4f})',
                 color=model_colors[model_name])
        col_index += 1
    
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title + ' - ' + model_name)
        plt.legend(loc='lower left', fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.7)

# Set the figure size
plt.figure(figsize=(20, 20))

# 1. Exercise classification PR curves
exercise_y_true = {
    'OpenAI': exercise_results['OpenAI']['y_true'],
    'Azure OpenAI': exercise_results['Azure OpenAI']['y_true'],
    'Bedrock': exercise_results['Bedrock']['y_true'],
}
exercise_preds = {
    'OpenAI': exercise_results['OpenAI']['y_scores'],
    'Azure OpenAI': exercise_results['Azure OpenAI']['y_scores'],
    'Bedrock': exercise_results['Bedrock']['y_scores'],
}
plot_pr_curves(exercise_y_true, exercise_preds, 
               'Exercise Classification: Precision-Recall Curve', 1)

# 2. Exercise duration PR curves
time_subset = data[data['duration_estimation'] != True]
time_y_true = {
    'OpenAI': time_results['OpenAI']['y_true'],
    'Azure OpenAI': time_results['Azure OpenAI']['y_true'],
    'Bedrock': time_results['Bedrock']['y_true']
}
time_preds = {
    'OpenAI': time_results['OpenAI']['y_scores'],
    'Azure OpenAI': time_results['Azure OpenAI']['y_scores'],
    'Bedrock': time_results['Bedrock']['y_scores']
}
plot_pr_curves(time_y_true, time_preds,
               'Duration Prediction: Precision-Recall Curve', 2)

# 3. Exercise calorie PR curves
calorie_subset = data[data['calorie_estimation'] != True]
calorie_y_true = {
    'OpenAI': calorie_results['OpenAI']['y_true'],
    'Azure OpenAI': calorie_results['Azure OpenAI']['y_true'],
    'Bedrock': calorie_results['Bedrock']['y_true']
}
calorie_preds = {
    'OpenAI': calorie_results['OpenAI']['y_scores'],
    'Azure OpenAI': calorie_results['Azure OpenAI']['y_scores'],
    'Bedrock': calorie_results['Bedrock']['y_scores']
}
plot_pr_curves(calorie_y_true, calorie_preds,
               'Calorie Prediction: Precision-Recall Curve', 3)

plt.tight_layout()
plt.show()
