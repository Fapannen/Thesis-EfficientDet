"""
Parsing script for efficientdet log.

Expects path to logfile as argument
"""
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import math

##################################################################
##                                                              ##
##                                                              ##
##                         UTILS                                ##
##                                                              ##
##                                                              ##
##################################################################

SEP='----------------------------------------------------------------------------'
JSON_FILENAME='json_outputs_efficientdet.json'

# Parse memory entry from /proc/smaps output
def parse_Kb(string):
	return int(string.split()[1])

# Convert text outputs to arrays of floats
def load_outputs(outputs):
	out = []

	for output in outputs:
		output = output[2:-2]
		detection = [float(item) for item in output.replace(',', '').split()]
		out.append(detection)

	return out

# Retrieve MemoryEntry and InferenceEntry entries from log
def get_entries(file):
	entries = file.read().split(SEP)

	init_state = entries[0]
	inferences = entries[1:]

	init_memory = []
	inference_entries = []

	for mem in init_state.split('\n\n')[:-1]:
		init_memory.append(MemoryEntry(mem))

	for inference in inferences:
		inference_entries.append(InferenceEntry(inference))

	return init_memory, inference_entries

def make_unique(outputs):
	uniques = []
	for output in outputs:
		if output[5] > 0 and output not in uniques:
			uniques.append(output)
		else:
			break

	return uniques

# Build JSON in order to evaluate on COCO dataset
def build_json(inferenceEntries, model, filename=JSON_FILENAME):
	json_ = []
	test_dev = json.load(open('image_info_test-dev2017.json', 'r'))
	for inference in inferenceEntries:
		found = False
		imgwidth = 0
		imgheight = 0
		for img in test_dev['images']:
			if int(img['id']) == int(inference.image):
				found = True
				imgwidth = int(img['width'])
				imgheight = int(img['height'])
				#print("Found ID ", inference.image)
				break

		if not found:
			print("ID ", inference.image, " not found")
			continue

		for output in inference.unique_output:
			temp = {
				'image_id'    : inference.image,
				'category_id' : int(output[6]),
				'bbox'        : [output[2] * (imgwidth / model),
								 output[1] * (imgheight / model),
								 (output[4] - output[2]) * (imgwidth / model),
								 (output[3] - output[1]) * (imgheight / model)],
				'score'       : float(output[5])
			}

			json_.append(temp)

	out = json.dumps(json_)
	with open(filename, 'w') as f:
		f.write(out)

def get_id(image_number):
	zeros = True
	res = ''
	for num in image_number:
		if num == '0':
			if zeros:
				continue
		zeros = False
		res += num

	return int(res)


def overall_stats(inferences):
	acc_memory = 0
	min_memory = math.inf
	max_memory = 0

	acc_detections = 0
	min_detections = math.inf
	max_detections = 0

	acc_inference_time = 0
	min_inference_time = math.inf
	max_inference_time = 0

	for inf in inferences:
		acc_memory += inf.memory.rss
		min_memory =  inf.memory.rss if inf.memory.rss < min_memory else min_memory
		max_memory =  inf.memory.rss if inf.memory.rss > max_memory else max_memory

		acc_detections += inf.num_detections
		min_detections =  inf.num_detections if inf.num_detections < min_detections else min_detections
		max_detections =  inf.num_detections if inf.num_detections > max_detections else max_detections

		acc_inference_time += inf.timems
		min_inference_time =  inf.timems if inf.timems < min_inference_time else min_inference_time
		max_inference_time =  inf.timems if inf.timems > max_inference_time else max_inference_time

	avg_memory         = acc_memory / len(inferences)
	avg_detections     = acc_detections / len(inferences) 
	avg_inference_time = acc_inference_time / len(inferences)

	print()
	print('Memory summary')
	print('--------------')
	print('Average memory usage: ', avg_memory)
	print('Minimum memory usage: ', min_memory)
	print('Maximum memory usage: ', max_memory)
	print()

	print('Detections summary')
	print('------------------')
	print('Average number of detections: ', avg_detections)
	print('Minimum number of detections: ', min_detections)
	print('Maximum number of detections: ', max_detections)
	print()

	print('Inference time summary')
	print('----------------------')
	print('Average inference time (ms): ', avg_inference_time)
	print('Minimum inference time (ms): ', min_inference_time)
	print('Maximum inference time (ms): ', max_inference_time)
	print()

def model_size_from_memory(init_memory):
	print('Model size: ', init_memory[1].rss - init_memory[0].rss)
	print('Model size after allocating tensors: ', init_memory[2].rss - init_memory[1].rss)
	print()

##################################################################
##                                                              ##
##                                                              ##
##                    CLASS DEFINITIONS                         ##
##                                                              ##
##                                                              ##
##################################################################


class MemoryEntry:
	def __init__(self, content):
		split = content.split('\n')
		if len(split) == 22:
			self.description     = split[0]
			self.misc            = split[1]
			self.rss             = parse_Kb(split[2])
			self.pss             = parse_Kb(split[3])
			self.pss_anon        = parse_Kb(split[4])
			self.pss_file        = parse_Kb(split[5])
			self.pss_shmem       = parse_Kb(split[6])
			self.shared_clean    = parse_Kb(split[7])
			self.shared_dirty    = parse_Kb(split[8])
			self.private_clean   = parse_Kb(split[9])
			self.private_dirty   = parse_Kb(split[10])
			self.referenced      = parse_Kb(split[11])
			self.anonymous       = parse_Kb(split[12])
			self.lazyfree        = parse_Kb(split[13])
			self.anonhugepages   = parse_Kb(split[14])
			self.shmempmdmapped  = parse_Kb(split[15])
			self.filepmdmapped   = parse_Kb(split[16])
			self.shared_hugetlb  = parse_Kb(split[17])
			self.private_hugetlb = parse_Kb(split[18])
			self.swap            = parse_Kb(split[19])
			self.swappss         = parse_Kb(split[20])
			self.locked          = parse_Kb(split[21])
		else:
			self.description     = 'Inference memory footprint'
			self.misc            = split[0]
			self.rss             = parse_Kb(split[1])
			self.pss             = parse_Kb(split[2])
			self.pss_anon        = parse_Kb(split[3])
			self.pss_file        = parse_Kb(split[4])
			self.pss_shmem       = parse_Kb(split[5])
			self.shared_clean    = parse_Kb(split[6])
			self.shared_dirty    = parse_Kb(split[7])
			self.private_clean   = parse_Kb(split[8])
			self.private_dirty   = parse_Kb(split[9])
			self.referenced      = parse_Kb(split[10])
			self.anonymous       = parse_Kb(split[11])
			self.lazyfree        = parse_Kb(split[12])
			self.anonhugepages   = parse_Kb(split[13])
			self.shmempmdmapped  = parse_Kb(split[14])
			self.filepmdmapped   = parse_Kb(split[15])
			self.shared_hugetlb  = parse_Kb(split[16])
			self.private_hugetlb = parse_Kb(split[17])
			self.swap            = parse_Kb(split[18])
			self.swappss         = parse_Kb(split[19])
			self.locked          = parse_Kb(split[20])

class InferenceEntry:
	def __init__(self, content):
		split = content.split('\n\n')

		memory_footprint = split[0]
		inference_log    = split[1].split('\n')
		
		self.memory         = MemoryEntry(memory_footprint)
		self.image          = get_id(inference_log[0].split('/')[1].split('.jpg')[0])
		self.timems         = int(inference_log[1].split(': ')[1])
		self.times          = self.timems / 1000
		self.output         = load_outputs(inference_log[2:-1])
		self.unique_output  = make_unique(self.output)
		self.num_detections = len(self.unique_output)
		
##################################################################
##                                                              ##
##                                                              ##
##                    MAIN                                      ##
##                                                              ##
##                                                              ##
##################################################################

logpath = sys.argv[1]
model   = int(sys.argv[2])

with open(logpath, 'r') as logfile:
	init_memory, inference_entries = get_entries(logfile)

	print('Creating JSON for coco evaluation ...')
	build_json(inference_entries, model, JSON_FILENAME)
	print('JSON created ...')

	print('Computing overall results')
	model_size_from_memory(init_memory)
	overall_stats(inference_entries)
