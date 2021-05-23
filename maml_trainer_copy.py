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

class Plotter(object):

	def __init__(self, name):
		self.name = name
		self.logger = defaultdict(lambda: []) #{'train': [], 'valid': []}

	def update(self, dict_):
		for k, v in dict_.items():
			self.logger[k].append(v)

	def plot(self):
		os.makedirs('./figs', exist_ok=True)
		for k, v in self.logger.items():
			iters = range(len(self.logger[k]))
			plt.plot(iters, self.logger[k], c='dodgerblue', label="k")
			plt.xlabel('epoch', fontsize=12)
			plt.ylabel(k, fontsize=12)
			plt.tight_layout()
			plt.savefig('./figs/' + self.name + '_ ' + k + '.png')


class Sampler(object):
	 
	def __init__(self, support_datasets, query_datasets):
		self.support_datasets = support_datasets
		self.query_datasets = query_datasets
		self.n_tasks = len(self.support_datasets)
		self.support_loaders = defaultdict(lambda: [])
		self.query_loaders = defaultdict(lambda: [])
		for task in range(self.n_tasks):
			self.support_loaders[task] = self._initialize_loaders(task, type_="support")
			self.query_loaders[task] = self._initialize_loaders(task, type_="query")

		self.exhausted = {k: {"support" : False, "query": False} for k in range(self.n_tasks)}
	
	def _initialize_loaders(self, task, type_="support"):
		if type_ == "support":
			return self.support_datasets[task].get_dataloaders()
		elif type_ == "query":
			return self.query_datasets[task].get_dataloaders()

	def set_exhausted(self, val, task, type_):
		self.exhausted[task][type_] = val
	
	def sample_support(self, task):
 
		dl = self.support_loaders[task]
		source_task_batch = {}
		for i, loader in enumerate(dl):
			b = next(loader, -1)
			if not b == -1:
				label = b["labels"][0].item()
				source_task_batch[label] = b
			else:
				self.set_exhausted(True, task, type_="support")
		return source_task_batch
	
	def sample_query(self, task):
		dl = self.query_loaders[task]
		query_task_batch = {}
		for i, loader in enumerate(dl):
			b = next(loader, -1)
			if not b == -1:
				label = b["labels"][0].item()
				query_task_batch[label] = b
			else:
				self.set_exhausted(True, task, type_="query")
		return query_task_batch

	def reset_sampler(self, task, type_):
		self.set_exhausted(False, task, type_)
		if type_ == "support":
			self.support_loaders[task] = self._initialize_loaders(task, type_="support")
		if type_ == "query":
			self.query_loaders[task] = self._initialize_loaders(task, type_="query")


class MetaTrainer(object):


	def __init__(self, model, train_sampler,
				valid_sampler, test_sampler,
				task_classes, epochs,
				inner_lr, outer_lr,
				n_inner_steps, num_episodes,
				model_save_path, results_save_path,
				clip_value, exp_name,
				seed=42, device=torch.device("cpu")):
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
		
		
		self.outer_optimizer = AdamW(self.outer_model.encoder.parameters(),
									self.outer_lr,  weight_decay=1e-4)
		self.outer_lr_scheduler = get_cosine_schedule_with_warmup(self.outer_optimizer, num_training_steps=self.n_epochs,  num_warmup_steps=int(0.10*self.n_epochs))

		self.inner_results = {"losses":defaultdict(list),
					"accuracy":defaultdict(list)}
		self.outer_results = {"losses":defaultdict(list),
					"accuracy":defaultdict(list)}
		self.test_results = {"losses":defaultdict(list),
					"accuracy":defaultdict(list)}

		# self.plotter = Plotter(self.exp_name)

	def set_seed(self, seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)

	def dump_results(self):
		with open(os.path.join(self.results_save_path, self.exp_name + '.txt'), 'w') as f:
			json.dump({"inner":self.inner_results,
				"outer": self.outer_results,
				"test": self.test_results}, f)

	def train(self, test_every=1):
		"""Run episodes and perform outerloop updates """
		for epoch in range(self.n_epochs):
			self.outer_optimizer.zero_grad()
			#for each of the task in the number of meta-training tasks
			for episode in range(self.num_episodes['train']):
				print("---- Starting episode {} of epoch {} ----".format(episode, epoch))
				support_set = self.train_sampler.sample_support(episode)
				if self.train_sampler.exhausted[episode]["support"]:
					self.train_sampler.reset_sampler(episode, type_="support")
					support_set = self.train_sampler.sample_support(episode)

				query_set = self.train_sampler.sample_query(episode)
				if self.train_sampler.exhausted[episode]["query"]:
					self.train_sampler.reset_sampler(episode, type_="query")
					query_set = self.train_sampler.sample_query(episode)

				self.train_episode(support_set, query_set, episode)

			if epoch % test_every == 0:
				test_loss, test_acc = self.validate("valid")
				print("Test performance: epoch {}, task {}, loss: {}, accuracy: {}".format(
						epoch, episode, test_loss, test_acc))
				if test_acc > self.BEST_VAL_ACC:
					self.BEST_VAL_ACC = test_acc
					torch.save(self.outer_model.state_dict(), os.path.join(self.model_save_path, self.exp_name + '.pt'))


			self.outer_optimizer.step()
			self.outer_lr_scheduler.step()
			self.dump_results()
		
		# self.plotter.plot()


	def forward(self, model, batch):
		logits = model(self._to_device(batch['input_ids']),
					token_type_ids=self._to_device(batch['token_type_ids']),
					attention_mask=self._to_device(batch['attention_mask']))
		return logits

	def _to_device(self, inp):
		if not torch.is_tensor(inp):
			inp = torch.tensor(inp)
		return inp.to(self.device)

	def init_prototype_parameters(self, model, n_classes, support_set, task):
		# print("In Init Prototypes ", task, n_classes)
		# print(support_set)
		prototypes = self._to_device(torch.zeros((n_classes, 768)))
		class_samples = self._to_device(torch.zeros((n_classes,1)))
		for label in range(n_classes):
			batches = support_set[label]
			# Batch is either a list of dicts, or a single dict.
			if isinstance(batches, dict):
				batches = [batches]

			n_batches = len(batches)
			for batch in batches:
				class_samples[label,:] += batch['input_ids'].size(0)
				input_ids = self._to_device(batch['input_ids'])
				token_type_ids = self._to_device(batch['token_type_ids'])
				attention_mask = self._to_device(batch['attention_mask'])

				encoding = self.outer_model.encoder(input_ids,
								token_type_ids=token_type_ids,
								attention_mask=attention_mask)["last_hidden_state"][:,0,:]
				prototypes[label, :] = prototypes[label, :] + self._to_device(torch.sum(encoding, dim=0))

		prototypes = prototypes / class_samples
		norm = prototypes.norm(p=2, dim=1, keepdim=True) #do L2 normalization
		model.gamma = prototypes.div(norm)
		

	def _extract(self, batch):
		shape = [batch[class_]["input_ids"].shape[1] for class_ in batch.keys()]
		max_shape = max(shape)
		input_ids = torch.cat(tuple([self._pad(batch[class_]["input_ids"], max_shape) for class_ in batch.keys()]), dim=0)
		token_type_ids = torch.cat(tuple([self._pad(batch[class_]["token_type_ids"], max_shape) for class_ in batch.keys()]), dim=0)
		attention_mask = torch.cat(tuple([self._pad(batch[class_]["attention_mask"], max_shape) for class_ in batch.keys()]), dim=0)
		labels = torch.cat(tuple([batch[class_]["labels"] for class_ in batch.keys()]))

		shuffle_indices = torch.randperm(labels.shape[0])
		return {'input_ids': input_ids[shuffle_indices],
				'token_type_ids': token_type_ids[shuffle_indices],
				'attention_mask': attention_mask[shuffle_indices],
				'labels': labels[shuffle_indices]
				}

	def _pad(self, tensor, max_shape):
		tensor = torch.nn.functional.pad(tensor, (0, max_shape - tensor.shape[1]), mode='constant', value=PAD_ID).detach()
		return tensor


	def inner_loop(self, model, n_classes, support_set, task):

		inner_loss = []
		inner_acc = []
		loss_func = nn.CrossEntropyLoss()
		model.zero_grad()
		optimizer = AdamW(model.parameters(), lr=self.inner_lr, weight_decay=1e-4)

		support_samples = self._extract(support_set)
		support_len = len(support_samples['labels'])
		for _ in range(self.n_inner_steps):
			batch_idx = np.linspace(0, support_len, self.n_inner_steps + 1, dtype=int)
			#print(batch_idx)
			for i, start_idx in enumerate(batch_idx[:-1]):
				 batch = {k:s[start_idx:batch_idx[i+1]] for k, s in support_samples.items()}
				 labels = self._to_device(batch['labels'])
				 optimizer.zero_grad()
				 logits = self.forward(model, batch)
				 loss = loss_func(logits, labels)
				 loss.backward()
				 torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
				 optimizer.step()

	def get_accuracy(self, logits, labels):
		#print(logits, labels)
		predictions = torch.argmax(logits, dim=1)
		return (predictions == labels).float().mean().item()


	def calc_validation_grads(self, model, n_classes, query_set, task):
		loss_func = nn.CrossEntropyLoss()

		batch = self._extract(query_set)
		labels = self._to_device(batch["labels"])
		logits = self.forward(model, batch)

		loss = loss_func(logits, labels)
		accuracy = self.get_accuracy(logits, labels)
		print("task: {}, query accuracy: {}, query loss: {}".format(task, accuracy, loss.item()))
		self.outer_results["losses"][task].append(loss.item())
		self.outer_results["accuracy"][task].append(accuracy)

		grads_inner_model = torch.autograd.grad(outputs=loss,
											inputs=model.encoder.parameters(),
											retain_graph=True,
											create_graph=True,
											allow_unused=True)

		grads_outer_model = torch.autograd.grad(outputs=loss,
											inputs=self.outer_model.encoder.parameters(),
											allow_unused=True)

		for i, (name, param) in enumerate(self.outer_model.named_parameters()):
			if 'pooler' in name:
				continue
			elif param.grad is None:
				param.grad = grads_inner_model[i] + grads_outer_model[i]
			else:
				param.grad += grads_inner_model[i] + grads_outer_model[i]


	def validate(self, mode, m=5):
		if mode == "valid":
			sampler = self.valid_sampler
		if mode == "test":
			sampler = self.test_sampler
		for task in range(self.num_episodes[mode]):
			n_classes = self.task_classes[mode][task]
			loss_func = nn.CrossEntropyLoss()
			inner_model = copy.deepcopy(self.outer_model)

			support_set = sampler.sample_support(task)
			if sampler.exhausted[task]["support"]:
					sampler.reset_sampler(task, type_="support")
					support_set = sampler.sample_support(task)
			for _ in range(m):
				self.init_prototype_parameters(inner_model, n_classes, support_set, task)
				inner_model.init_phi(n_classes)
				self.inner_loop(inner_model, n_classes, support_set, task)

			#validate on the whole query loader
			losses, accuracies = [], []
			with torch.no_grad():
					while True:
						query_set = sampler.sample_query(task)
						if sampler.exhausted[task]["query"]:
							break
						batch = self._extract(query_set)
						labels = self._to_device(batch["labels"])
						logits = self.forward(inner_model, batch)
						losses.append(loss_func(logits, labels).item())
						accuracies.append(self.get_accuracy(logits, labels))
					sampler.reset_sampler(task, type_="query")

			avg_loss = np.mean(losses)
			avg_acc = np.mean(accuracies)
			self.test_results["losses"][task].append(avg_loss)
			self.test_results["accuracy"][task].append(avg_acc)

		# self.plotter.update({"loss" : avg_loss, "accuracy": avg_acc})
		return avg_loss, avg_acc

	def train_episode(self, support_set, query_set, task):
		"train inner model for 1 step, returns gradients of encoder on support set."
		n_classes = self.task_classes['train'][task]
		loss_func = nn.CrossEntropyLoss()

		# Step 2: Duplicate model
		inner_model = copy.deepcopy(self.outer_model)

		# Step 3: Init prototype vectors (for now just take n embedding vectors).
		print("---- Initializing prototype parameters ----")
		self.init_prototype_parameters(inner_model, n_classes, support_set, task)

		# Step 4: Init output parameters (phi).
		inner_model.init_phi(n_classes)

		# Step 5: perform k inner loop steps.
		print("---- Performing inner loop updates ----")
		self.inner_loop(inner_model, n_classes, support_set, task)

		# Step 6: Replace output parameters with trick.
		inner_model.replace_phi()

		# Step 7: Apply trained model on query set.
		print("---- Calculating gradients on query set ----")
		self.calc_validation_grads(inner_model, n_classes, query_set, task)






if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--freeze_bert", action="store_true",
						help="Whether to freeze BERT parameters.")
	parser.add_argument("--epochs", type=int, default=50000,
						help="Number of outerloop updates to run.")
	parser.add_argument("--outer_lr", type=float, default=1e-4,
						help="learning rate for outer loop optimizer.")
	parser.add_argument("--inner_lr", type=float, default=1e-4,
						help="learning rate for inner loop optimizer.")
	parser.add_argument("--support_k", type=int, default=2,
						help="Number of support samples for each class.")
	parser.add_argument("--query_k", type=int, default=2,
						help="Number of query samples for each class.")
	parser.add_argument("--model_save_path", type=str, default="saved_models/",
						help="location to store saved model")
	parser.add_argument("--results_save_path", type=str, default="results/")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--clip_value", type=float, default=2.0)
	parser.add_argument("--device", default=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
	parser.add_argument("--n_inner_steps", type=int, default=5,
						help="number of batches for the inner_loop")
	parser.add_argument("--exp_name", default='default', type=str, help="Model and results will be saved here")



	config = parser.parse_args().__dict__


	#All the Tasks that are allowed are: MNLI(has dev set), Paraphrase, Stance, VitaminC(has dev set), SciTail(has dev set)
	def get_tasks(meta_training_tasks=['mnli','scitail'],
				meta_validation_tasks=['paraphrase'],
				meta_testing_tasks=['vitaminc'], slice=-1):
		train_support = []
		train_query = []
		train_classes = []
		for training_task in meta_training_tasks:
			support, query, classes = sample_metaset(config, training_task, 'train', slice=slice)
			train_support.append(support)
			train_query.append(query)
			train_classes.append(classes)
		
		valid_support = []
		valid_query = []
		valid_classes = []
		for valid_task in meta_validation_tasks:
			support, query, classes = sample_metaset(config, valid_task, 'valid', slice=slice)
			valid_support.append(support)
			valid_query.append(query)
			valid_classes.append(classes)

		test_support = []
		test_query = []
		test_classes = []
		for test_task in meta_testing_tasks:
			support, query, classes = sample_metaset(config, test_task, 'test', slice=slice)
			test_support.append(support)
			test_query.append(query)
			test_classes.append(classes)

		
		train_task_classes = {k:v for k,v in enumerate(train_classes)}
		valid_task_classes = {k:v for k,v in enumerate(valid_classes)}
		test_task_classes = {k:v for k,v in enumerate(test_classes)}
		task_classes = (train_task_classes, valid_task_classes, test_task_classes)
		return Sampler(train_support, train_query), Sampler(valid_support, valid_query), Sampler(test_support, test_query), task_classes

	model = Classifier(config)
	train_sampler, valid_sampler, test_sampler, task_classes = get_tasks(slice=1000)


	meta_trainer = MetaTrainer(
							model = model,
							train_sampler = train_sampler,
							valid_sampler = valid_sampler,
							test_sampler = test_sampler,
							task_classes = {'train': task_classes[0], 'valid': task_classes[1], 'test': task_classes[2]},
							epochs = config["epochs"],
							inner_lr = config['inner_lr'],
							outer_lr = config['outer_lr'],
							n_inner_steps = config["n_inner_steps"],
							num_episodes = {'train': len(task_classes[0].keys()), 'valid': len(task_classes[1].keys()), 'test': len(task_classes[2].keys())},
							model_save_path = config["model_save_path"],
							results_save_path = config["results_save_path"],
							device = config["device"],
							clip_value = config["clip_value"],
							exp_name = config["exp_name"],
							seed = config["seed"]
							)

	meta_trainer.train(test_every=1)
