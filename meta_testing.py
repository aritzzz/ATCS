import copy
import torch.nn as nn
import json
import argparse
from tqdm import tqdm
from data_utils import *
from models import Classifier
from collections import defaultdict
import matplotlib.pyplot as plt
from transformers import AdamW, get_cosine_schedule_with_warmup
import torch
import os
import numpy as np
from maml_trainer_copy import *

class MetaTester(MetaTrainer):
	def __init__(self, model, train_sampler,
				valid_sampler, test_sampler,
				task_classes, epochs,
				inner_lr, outer_lr,
				n_inner_steps, num_episodes,
				model_save_path, results_save_path,
				clip_value, exp_name,
				seed=42, device=torch.device("cpu")):
		super(MetaTrainer, self).__init__()
		self.set_seed(seed)
		self.outer_model = model.to(device)
		self.n_tasks = num_episodes
		self.n_epochs = epochs
		self.exp_name = exp_name
		self.inner_lr = inner_lr
		self.outer_lr = outer_lr
		self.n_inner_steps = n_inner_steps
		self.num_episodes = num_episodes
		self.device = device
		self.task_classes = task_classes
		self.clip_value = clip_value
		
		self.model_save_path = model_save_path
		self.results_save_path = results_save_path
		
		os.makedirs(self.model_save_path, exist_ok = True)
		os.makedirs(self.results_save_path, exist_ok = True)
		
		self.train_sampler = train_sampler
		self.valid_sampler = valid_sampler
		self.test_sampler = test_sampler

		self.BEST_VAL_ACC = 0.0
		
		print(self.num_episodes)
		print(self.task_classes)

		self.loss_funcs = {
					0: nn.CrossEntropyLoss(),
					1: nn.CrossEntropyLoss(),
					2: nn.CrossEntropyLoss()
				}
		
		
		

		self.inner_results = {"losses":defaultdict(list),
					"accuracy":defaultdict(list)}
		self.outer_results = {"losses":defaultdict(list),
					"accuracy":defaultdict(list)}
		self.test_results = {"losses":defaultdict(list),
					"accuracy":defaultdict(list)}

	def test(self):
		for episode in range(self.num_episodes['test']):
			self.validate(episode)

		self.dump_results()

	def dump_results(self):
		with open(os.path.join(self.results_save_path, self.exp_name + '_testing.txt'), 'w') as f:
			json.dump({"test": self.test_results}, f)

	def validate(self, task, K=[2,4,8]):
		n_classes = self.task_classes['test'][task]
		loss_func = nn.CrossEntropyLoss()

		losses_ = []
		accuracies_ = []
		for i in K:
			inner_model = copy.deepcopy(self.outer_model)
			j = 0
			while j < i:
				print(j,i)
				support_set = self.test_sampler.sample_support(task)
				if self.test_sampler.exhausted[task]["support"]:
						self.test_sampler.reset_sampler(task, type_="support")
						support_set = self.test_sampler.sample_support(task)
				for _ in range(self.n_inner_steps):
					self.init_prototype_parameters(inner_model, n_classes, support_set, task)
					inner_model.init_phi(n_classes)
					self.inner_loop(inner_model, n_classes, support_set, task)

				#validate on the whole query loader
				losses, accuracies = [], []
				with torch.no_grad():
						while True:
							query_set = self.test_sampler.sample_query(task)
							if self.test_sampler.exhausted[task]["query"]:
								break
							batch = self._extract(query_set)
							labels = self._to_device(batch["labels"])
							logits = self.forward(inner_model, batch)
							losses.append(loss_func(logits, labels).item())
							accuracies.append(self.get_accuracy(logits, labels))
						self.test_sampler.reset_sampler(task, type_="query")
				losses_.append(np.mean(losses))
				accuracies_.append(np.mean(accuracies))
				j+=1

		avg_loss = np.mean(losses_)
		avg_acc = np.mean(accuracies_)
		self.test_results["losses"][task].append(avg_loss)
		self.test_results["accuracy"][task].append(avg_acc)
		#self.plotter.update({"loss" : avg_loss, "accuracy": avg_acc})
		return avg_loss, avg_acc		






if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_save_path', type=str, default='./Models',
					help='path to save the checkpoints')
	parser.add_argument('--exp_name', type=str, default='default',
					help='Name of the experiment. Checkpoints will be saved with this name')
	parser.add_argument('--device', type=str, default=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
					help='Device to use cpu, or cuda')
	parser.add_argument("--outer_lr", type=float, default=1e-4,
						help="learning rate for outer loop optimizer.")
	parser.add_argument("--inner_lr", type=float, default=1e-4,
						help="learning rate for inner loop optimizer.")
	parser.add_argument("--support_k", type=int, default=2,
						help="Number of support samples for each class.")
	parser.add_argument("--query_k", type=int, default=2,
						help="Number of query samples for each class.")
	parser.add_argument("--n_inner_steps", type=int, default=5)
	parser.add_argument("--clip_value", type=float, default=1.5)
	parser.add_argument("--results_save_path", type=str, default='default')
	parser.add_argument("--seed", type=int, default=42)

	config = parser.parse_args().__dict__

	config2 = {'freeze_bert': False}
	model = Classifier(config2)
	checkpoint_path = os.path.join(config['model_save_path'], config['exp_name'] + '.pt')
	checkpoint = torch.load(checkpoint_path, map_location=torch.device(config['device']))
	model.load_state_dict(checkpoint)

	def get_meta_testing_dataset(meta_testing_tasks=['mnli', 'stance', 'scitail', 'paraphrase'], slice=-1):
		test_support = []
		test_query = []
		test_classes = []
		for test_task in meta_testing_tasks:
			support, query, classes = sample_metaset(config, test_task, 'test', slice=slice)
			test_support.append(support)
			test_query.append(query)
			test_classes.append(classes)

		test_task_classes = {k:v for k,v in enumerate(test_classes)}

		return Sampler(test_support, test_query), test_task_classes


	test_sampler, test_task_classes = get_meta_testing_dataset()

	meta_tester = MetaTester(
							model = model,
							train_sampler = None,
							valid_sampler = None,
							test_sampler = test_sampler,
							task_classes = {'train': None, 'valid': None, 'test': test_task_classes},
							epochs = None,
							inner_lr = config['inner_lr'],
							outer_lr = config['outer_lr'],
							n_inner_steps = config["n_inner_steps"],
							num_episodes = {'train': None, 'valid': None, 'test': len(test_task_classes.keys())},
							model_save_path = config["model_save_path"],
							results_save_path = config["results_save_path"],
							device = config["device"],
							clip_value = config["clip_value"],
							exp_name = config["exp_name"],
							seed = config["seed"]
							)

	meta_tester.test()