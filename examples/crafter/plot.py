import matplotlib.pyplot as plt
import numpy as np


# 2M
rewards = np.array([13.73, 15.58, 15.26, 12.62, 9.88])
surtime = np.array([325.24, 356.1, 363.82, 308.69, 269.39])
data_x = surtime / 353.19 / 2 + rewards / 15.32 / 2
data_y = np.array([81.6, 82.56, 87.52, 96.28, 87.0])
name = ["0.1", "0.3", "0.5", "1.0", "inf"]


plt.scatter(data_x, data_y)
for i in range(len(data_x)):
    plt.text(data_x[i], data_y[i], name[i])

# survival time-step 353.72   348.9
# original_rewards 15.64   15.219999999999995
# instruction following rewards 83.2   26.52
# model models/crafter_agent-10M-49/

# survival time-step 316.76   297.64
# original_rewards 14.289999999999996   11.469999999999995                                                       
# instruction following rewards 94.16   36.36
# model models/crafter_agent-10M-50/

# survival time-step 293.87   275.97
# original_rewards 11.24   8.52
# instruction following rewards 110.48   41.96
# model models/crafter_agent-10M-51/

# survival time-step 269.32   243.69
# original_rewards 9.340000000000002   5.68
# instruction following rewards 97.52   26.88
# model models/crafter_agent-10M-52/

# survival time-step 246.62   232.89
# original_rewards 8.720000000000002   6.08
# instruction following rewards 103.28   27.76
# model models/crafter_agent-10M-53/

# survival time-step 250.33   249.78
# original_rewards 9.490000000000002   6.53
# instruction following rewards 102.0   33.12
# model models/crafter_agent-10M-54/
    
# 10M
rewards = np.array([15.64, 14.28, 11.24, 9.34, 8.72, 9.49])
surtime = np.array([353.72, 316.76, 293.87, 269.32, 246.62, 250.33])
data_x = surtime / 353.19 / 2 + rewards / 15.32 / 2
data_y = np.array([83.2, 94.16, 110.48, 97.52, 103.28, 102.0])
name = ["0.1", "0.5", "1.0", "2.0", "4.0", "inf"]
plt.scatter(data_x, data_y, c="red")
for i in range(len(data_x)):
    plt.text(data_x[i], data_y[i], name[i])

plt.legend(["2M", "10M"])
plt.xlabel("Normalized rewards")
plt.ylabel("Instruction following rewards")
plt.title("fine-tuning")
plt.savefig("run_results/crafter.png")
