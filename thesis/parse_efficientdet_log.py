"""
Parsing script for efficientdet log.

Expects path to logfile as argument
"""
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

##################################################################
##                                                              ##
##                                                              ##
##                         UTILS                                ##
##                                                              ##
##                                                              ##
##################################################################

SEP='----------------------------------------------------------------------------'

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

# Build JSON in order to evaluate on COCO dataset
def build_json(inferenceEntries, filename):
	pass



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
		
		self.memory = MemoryEntry(memory_footprint)
		self.image  = inference_log[0].split('/')[1]
		self.timems = int(inference_log[1].split(': ')[1])
		self.times  = self.timems / 1000
		self.output = load_outputs(inference_log[2:-1])
		self.iou    = None
		self.map    = None
		print(self.memory.rss)

##################################################################
##                                                              ##
##                                                              ##
##                    MAIN                                      ##
##                                                              ##
##                                                              ##
##################################################################

logpath = sys.argv[1]

with open(logpath, 'r') as logfile:
	init_memory, inference_entries = get_entries(logfile)
