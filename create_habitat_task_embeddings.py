import os
from collections import OrderedDict
import json

import torch
from transformers import CLIPTokenizer, CLIPTextModel


def save_dict_as_json(data, filepath):
	# Convert any torch.Tensors to Python scalars/lists
	def convert(obj):
		if isinstance(obj, torch.Tensor):
			return obj.tolist()
		if isinstance(obj, dict):
			return {k: convert(v) for k, v in obj.items()}
		if isinstance(obj, list):
			return [convert(i) for i in obj]
		return obj

	with open(filepath, 'w') as f:
		json.dump(convert(data), f, indent=2)


# Embodiment descriptions
EMB_HABITAT = "2D object goal navigation agent with 6 actions: stop, move forward, turn left, turn right, look up, look down"

# Task descriptions
TASKS = OrderedDict({
    "counter": {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a counter',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    "fireplace": {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a fireplace',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    "gym_equipment": {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to gym equipment',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    "clothes": {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to clothes',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'plant': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a plant',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'sink': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a sink',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'toilet': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a toilet',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'stool': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a stool',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'towel': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a towel',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'tv_monitor': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a tv monitor',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'shower': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a shower',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'bathtub': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a bathtub',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'picture': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a picture',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'cabinet': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a cabinet',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'cushion': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a cushion',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'sofa': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a sofa',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'bed': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a bed',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'chest_of_drawers': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a chest of drawers',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'table': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a table',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'chair': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a chair',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
    'seating': {
        'embodiment': EMB_HABITAT,
        'instruction': 'Navigate to a seating area',
        'action_dim': 6,
        'discrete_actions': True,
        'max_episode_steps': 500,
    },
})


# Load CLIP tokenizer and text model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
text_model.eval()
text_model.to('cuda')

# Create embeddings for each task
print(f'Found {len(TASKS)} tasks. Creating text embeddings...')
for task_name, task_info in TASKS.items():
	# Embed object category from task key (underscores -> spaces)
	text = task_name.replace("_", " ")

	# Tokenize the task description
	inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
	inputs = {k: v.to('cuda') for k, v in inputs.items()}

	# Get the text embedding
	with torch.no_grad():
		text_embeddings = text_model(**inputs).last_hidden_state

	# Store the embedding in the task info
	task_info['text_embedding'] = text_embeddings.mean(dim=1).squeeze().cpu().numpy().tolist()

# Save the task dictionary as a JSON file
FILEPATH = './tasks.json'  # specify your desired path here
assert os.path.exists(os.path.dirname(FILEPATH)), f'Directory does not exist: {os.path.dirname(FILEPATH)}'
save_dict_as_json(TASKS, FILEPATH)
print(f'Saved task embeddings of dim {len(TASKS["table"]["text_embedding"])}.')
