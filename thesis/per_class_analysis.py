
# Names of directories under 'results' directory which to analyze
models_paths = ['efficientdet-d0', 'efficientdet-d2', 'efficientdet-lite0', 'efficientdet-lite2', 'efficientdet-lite3',
			'ssd_mobilenet_v2_320']

# Finds top5 and worst5 (category, values) in 'plot' dictionary
def find_min5_max5(plot):
	min5 = {}
	max5 = {}

	vals = sorted(plot.items(), key=lambda x:x[1])

	return (vals[:5], vals[-5:])

# Produce an csv file of a given plot
def build_csv(plot, out_filename):
	with open(out_filename, 'w') as out:

		out.write('model')

		# An arbitrary choice, get the categories
		for key in plot['efficientdet-d0'].keys():
			out.write(',' + key)

		out.write('\n')

		# build the actual content
		for model in models_paths:
			out.write(model)
			for category in plot[model]:
				out.write(',' + str(plot[model][category]))
			out.write('\n')

AP_plot       = {}
AP50_plot     = {}
AP75_plot     = {}
APsmall_plot  = {}
APmedium_plot = {}
APlarge_plot  = {}
ARmax1_plot   = {}
ARmax10_plot  = {}
ARmax100_plot = {}
ARsmall_plot  = {}
ARmedium_plot = {}
ARlarge_plot  = {}

for model_path in models_paths:

	AP_plot[model_path]       = {}
	AP50_plot[model_path]     = {}
	AP75_plot[model_path]     = {}
	APsmall_plot[model_path]  = {}
	APmedium_plot[model_path] = {}
	APlarge_plot[model_path]  = {}
	ARmax1_plot[model_path]   = {}
	ARmax10_plot[model_path]  = {}
	ARmax100_plot[model_path] = {}
	ARsmall_plot[model_path]  = {}
	ARmedium_plot[model_path] = {}
	ARlarge_plot[model_path]  = {}

	with open('results/' + model_path + '/scoring_output_log.txt', 'r') as f:

		categories = [content for content in f.read().split('evaluate category: ')[1:]]

		for cat in categories:
			lines = [line for line in cat.split('\n')]

			category_name = lines[0]

			values = lines[7:]

			AP       = float(values[0].split('=')[4])
			AP50     = float(values[1].split('=')[4])
			AP75     = float(values[2].split('=')[4])
			APsmall  = float(values[3].split('=')[4])
			APmedium = float(values[4].split('=')[4])
			APlarge  = float(values[5].split('=')[4])
			ARmax1   = float(values[6].split('=')[4])
			ARmax10  = float(values[7].split('=')[4])
			ARmax100 = float(values[8].split('=')[4])
			ARsmall  = float(values[9].split('=')[4])
			ARmedium = float(values[10].split('=')[4])
			ARlarge  = float(values[11].split('=')[4])

			AP_plot[model_path][category_name]       = AP
			AP50_plot[model_path][category_name]     = AP50
			AP75_plot[model_path][category_name]     = AP75
			APsmall_plot[model_path][category_name]  = APsmall
			APmedium_plot[model_path][category_name] = APmedium
			APlarge_plot[model_path][category_name]  = APlarge
			ARmax1_plot[model_path][category_name]   = ARmax1
			ARmax10_plot[model_path][category_name]  = ARmax10
			ARmax100_plot[model_path][category_name] = ARmax100
			ARsmall_plot[model_path][category_name]  = ARsmall
			ARmedium_plot[model_path][category_name] = ARmedium
			ARlarge_plot[model_path][category_name]  = ARlarge

for model_path in models_paths:
	print(model_path, ' summary:')
	print()

	# Get top5 and worst5 values and categories for Average Precision
	minT, maxT = find_min5_max5(AP_plot[model_path])
	print('Worst 5 AP performance (category, AP): ', minT)
	print('Best  5 AP performance (category, AP): ', maxT)
	print()

	# Get top5 and worst5 values and categories for Average Recall with 100 max detections
	minT, maxT = find_min5_max5(ARmax100_plot[model_path])
	print('Worst 5 ARmax100 performance (category, ARmax100): ', minT)
	print('Best  5 ARmax100 performance (category, ARmax100): ', maxT)
	print()
	print('---------------------------------------------------------------------------------')

build_csv(AP_plot,       'AP_table.csv')
build_csv(AP50_plot,     'AP50_table.csv')
build_csv(AP75_plot,     'AP75_table.csv')
build_csv(APsmall_plot,  'APsmall_table.csv')
build_csv(APmedium_plot, 'APmedium_table.csv')
build_csv(APlarge_plot,  'APlarge_table.csv')
build_csv(ARmax1_plot,   'ARmax1_table.csv')
build_csv(ARmax10_plot,  'ARmax10_table.csv')
build_csv(ARmax100_plot, 'ARmax100_table.csv')
build_csv(ARsmall_plot,  'ARsmall_table.csv')
build_csv(ARmedium_plot, 'ARmedium_table.csv')
build_csv(ARlarge_plot,  'ARlarge_table.csv')
