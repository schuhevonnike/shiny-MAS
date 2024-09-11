import pandas as pd

# Load the datasets
evaluation_data = pd.read_csv(r'/Users/Niklas/Library/CloudStorage/OneDrive-tukl/TU KL/02_Master/10. Fachsemester/[4.5 LP] Multiagent Systems/Programming Task/Konzept/project/evaluation_data/evaluation_data.csv')
training_data = pd.read_csv(r'/Users/Niklas/Library/CloudStorage/OneDrive-tukl/TU KL/02_Master/10. Fachsemester/[4.5 LP] Multiagent Systems/Programming Task/Konzept/project/evaluation_data/training_data.csv')
# Function to split the observation columns into separate columns
def split_observations(data, observation_col, prefix):
    # Strip the brackets, split by space or comma, and convert to a dataframe
    observations = data[observation_col].str.strip('[]').str.split(expand=True)
    # Rename columns with a prefix (Observation_1, Observation_2, ...)
    observations.columns = [f'{prefix}_{i+1}' for i in range(observations.shape[1])]
    return pd.concat([data.drop(columns=[observation_col]), observations], axis=1)

# Split 'Observation' and 'Next Observation' into separate columns for both datasets
evaluation_data = split_observations(evaluation_data, 'Observation', 'Observation')
evaluation_data = split_observations(evaluation_data, 'Next Observation', 'Next_Observation')

training_data = split_observations(training_data, 'Observation', 'Observation')
training_data = split_observations(training_data, 'Next Observation', 'Next_Observation')

# Save the reformatted datasets
evaluation_data.to_csv('reformatted_evaluation_data.csv', index=False)
training_data.to_csv('reformatted_training_data.csv', index=False)

print("Reformatted CSV files saved.")
