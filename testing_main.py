from main import all_history
import matplotlib.pyplot as plt

plt.xlabel("# Gen")
plt.ylabel("Loss Magnitude")
plt.plot(all_history.history["loss"])
