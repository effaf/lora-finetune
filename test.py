import transformers
print(transformers.__version__)

import transformers
print(transformers.__file__)

from transformers import TrainingArguments

args = TrainingArguments(output_dir="./test_output")
print(args)

from transformers import TrainingArguments
help(TrainingArguments)

