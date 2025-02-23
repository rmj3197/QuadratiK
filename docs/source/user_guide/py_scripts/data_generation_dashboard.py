# # Data Generation for Normality Test
import numpy as np
import pandas as pd

normal_samples = np.random.normal(loc=0, scale=1, size=(100,2))
np.savetxt('normal_samples.csv', normal_samples, delimiter=',', header='', comments='')


# # Data Generation for Two Sample Test
import numpy as np
import pandas as pd

sample1 = np.random.normal(loc=0, scale=1, size=(100,2))
np.savetxt('Sample_X.csv', sample1, delimiter=',', header='', comments='')

sample2 = np.random.normal(loc=0, scale=2, size=(100,2))
np.savetxt('Sample_Y.csv', sample2, delimiter=',', header='', comments='')


# # Data Generation for K-Sample Test
import numpy as np
import pandas as pd

sample1 = np.random.normal(loc=0, scale=1, size=(100,3))
sample2 = np.random.normal(loc=0.1, scale=1, size=(100,3))
sample3 = np.random.normal(loc=0.2, scale=1, size=(100,3))

sample1_labels = [1] * sample1.shape[0] 
sample2_labels = [2] * sample2.shape[0]    
sample3_labels = [3] * sample3.shape[0] 

stacked_samples = np.vstack((sample1, sample2, sample3))
stacked_labels = np.concatenate((sample1_labels, sample2_labels, sample3_labels))

stacked_samples_with_labels = np.column_stack((stacked_samples, stacked_labels))

np.savetxt('K_Sample_Data.csv', stacked_samples_with_labels, delimiter=',', header='', comments='')

# # Data Generation for Uniformity Test
data = np.random.normal(size=(200, 3))
data_unif = data / np.sqrt(np.sum(data**2, axis=1, keepdims=True))
np.savetxt('Data_Uniformity_Test.csv', data_unif, delimiter=',', header='', comments='')


# # Crabs Dataset for Clustering
import pandas as pd

# Load the crabs dataset from seaborn
crabs = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/refs/heads/master/csv/MASS/crabs.csv').drop(columns=['rownames', 'index', 'sex'])
# Map the 'sp' column to numbers
sp_mapping = {species: idx for idx, species in enumerate(crabs['sp'].unique())}
crabs['sp'] = crabs['sp'].map(sp_mapping)
# Save the crabs dataset to a CSV file
crabs.to_csv('crabs_dataset.csv', index=False)



