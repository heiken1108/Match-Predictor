import matplotlib.pyplot as plt
import numpy as np

anomaly_color = 'sandybrown'
prediction_color = 'yellowgreen'
training_color = 'yellowgreen'
validation_color = 'gold'
test_color = 'coral'
figsize=(9, 3)

def plot_series(data, x_values=None, labels=None,
										windows=None,
										predictions=None,
										highlights=None,
										val_start=None,
										test_start=None,
										threshold=None,
										figsize=figsize,
										xlabel=None,
										ylabel=None):
		# Open a new figure
		plt.close('all')
		plt.figure(figsize=figsize)

		if x_values == []:
			x = range(len(data))
		elif x_values is not None:
			x = x_values
		else:
			x = data.index
		# Plot data
		plt.plot(x, data.values, zorder=0)
		# Rotated x ticks
		plt.xticks(rotation=45)
		# Plot labels
		if labels is not None:
				plt.scatter(labels.values, data.loc[labels],
										color=anomaly_color, zorder=2)
		# Plot windows
		if windows is not None:
				for _, wdw in windows.iterrows():
						plt.axvspan(wdw['begin'], wdw['end'],
												color=anomaly_color, alpha=0.3, zorder=1)
		
		# Plot training data
		if val_start is not None:
				plt.axvspan(x[0], val_start,
										color=training_color, alpha=0.1, zorder=-1)
		if val_start is None and test_start is not None:
				plt.axvspan(x[0], test_start,
										color=training_color, alpha=0.1, zorder=-1)
		if val_start is not None:
				plt.axvspan(val_start, test_start,
										color=validation_color, alpha=0.1, zorder=-1)
		if test_start is not None:
				plt.axvspan(test_start, x[-1],
										color=test_color, alpha=0.3, zorder=0)
		# Predictions
		if predictions is not None:
				plt.scatter(predictions.values, data.loc[predictions],
										color=prediction_color, alpha=.4, zorder=3)
		# Plot threshold
		if threshold is not None:
				plt.plot([x[0], x[-1]], [threshold, threshold], linestyle=':', color='tab:red')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid(':')
		plt.tight_layout() 


def plot_histogram(y_values, figsize=figsize):
	bins = np.arange(y_values.min() - 0.5, y_values.max() + 1.5, 1)
	y_values.hist(bins=bins, figsize=figsize, edgecolor='black', rwidth=0.9)